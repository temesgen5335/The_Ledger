"""Extraction router — per-page strategy selection with confidence-gated escalation."""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ledger.agents.refinery.agents.page_analyzer import PageAnalyzer
from ledger.agents.refinery.models.document_profile import DocumentProfile
from ledger.agents.refinery.models.extracted_document import ExtractedDocument, ExtractedPage
from ledger.agents.refinery.strategies.fast_text import FastTextExtractor
from ledger.agents.refinery.strategies.layout import LayoutExtractor
from ledger.agents.refinery.strategies.vision import VisionExtractor


def _load_rules(rules_path: Path | None = None) -> dict:
    base = Path(__file__).resolve().parent.parent.parent
    if rules_path and rules_path.exists():
        with open(rules_path, "r") as f:
            return yaml.safe_load(f) or {}
    for candidate in [base / "rubric" / "extraction_rules.yaml", base / "extraction_rules.yaml"]:
        if candidate.exists():
            with open(candidate, "r") as f:
                return yaml.safe_load(f) or {}
    return {}


def _merge_extracted_documents(
    doc_id: str,
    profile: DocumentProfile,
    docs: list[ExtractedDocument],
) -> ExtractedDocument:
    """Merge pages/tables/figures from multiple partial ExtractedDocuments into one."""
    all_pages: list[ExtractedPage] = []
    all_tables = []
    all_figures = []
    strategies_used: set[str] = set()

    for d in docs:
        all_pages.extend(d.pages)
        all_tables.extend(d.tables)
        all_figures.extend(d.figures)
        strategies_used.add(d.strategy_used)

    all_pages.sort(key=lambda p: p.page_number)

    strategy_label = "mixed" if len(strategies_used) > 1 else next(iter(strategies_used), "fast_text")
    confidences = [p.confidence for p in all_pages if p.confidence is not None]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return ExtractedDocument(
        doc_id=doc_id,
        profile=profile,
        pages=all_pages,
        tables=all_tables,
        figures=all_figures,
        strategy_used=strategy_label,
        confidence_score=round(avg_conf, 4),
    )


class ExtractionRouter:
    """Two-level extraction router: document profiling sets a baseline, then
    page-level analysis (via precomputed pdfplumber stats) overrides individual
    pages to cheaper strategies when safe.

    Extraction results are cached to `cache_dir` (default: `.refinery/extractions/`)
    as ``{doc_id}.json`` so subsequent pipeline runs skip expensive re-extraction.
    Pass ``force_reextract=True`` to ``extract()`` to bypass the cache.
    """

    def __init__(
        self,
        rules_path: Path | None = None,
        ledger_path: Path | str | None = None,
        cache_dir: Path | str | None = None,
        stats_dir: Path | str | None = None,
        gemini_api_key: str | None = None,
    ):
        self._rules = _load_rules(rules_path)
        esc = self._rules.get("escalation", {})
        self.strategy_a_min = float(esc.get("strategy_a_min_confidence", 0.6))
        self.strategy_b_min = float(esc.get("strategy_b_min_confidence", 0.5))
        budget = self._rules.get("budget_guard", {})
        self.max_cost_per_doc = float(budget.get("max_cost_per_document_usd", 1.0))
        self.cost_per_image_usd = float(budget.get("cost_per_image_usd", 0.00035))
        self.cost_per_layout_page_usd = float(budget.get("cost_per_layout_page_usd", 0.0))
        self._on_budget_exceeded = str(budget.get("on_budget_exceeded", "use_strategy_b"))

        _base = Path(__file__).resolve().parent.parent.parent
        self._ledger_path = Path(ledger_path) if ledger_path else _base / ".refinery" / "extraction_ledger.jsonl"
        self._cache_dir = Path(cache_dir) if cache_dir else _base / ".refinery" / "extractions"
        self._fast = FastTextExtractor(
            min_chars_per_page=int(self._rules.get("triage", {}).get("min_chars_per_page", 100))
        )
        self._layout = LayoutExtractor()
        self._vision = VisionExtractor(
            max_cost_per_document_usd=self.max_cost_per_doc,
            cost_per_image_usd=self.cost_per_image_usd,
            api_key=gemini_api_key,
        )
        self._page_analyzer = PageAnalyzer(stats_dir=stats_dir, rules_path=rules_path)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, doc_id: str) -> Path:
        return self._cache_dir / f"{doc_id}.json"

    def _load_from_cache(self, doc_id: str) -> ExtractedDocument | None:
        path = self._cache_path(doc_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            doc = ExtractedDocument.model_validate(raw["doc"])
            print(
                f"[cache] HIT {doc_id} "
                f"(strategy={doc.strategy_used}, conf={doc.confidence_score:.2f}, "
                f"cached_at={raw.get('cached_at', 'unknown')})",
                file=sys.stderr,
            )
            return doc
        except Exception as exc:
            print(f"[cache] WARN corrupt cache for {doc_id}, ignoring: {exc}", file=sys.stderr)
            return None

    def _save_to_cache(self, doc: ExtractedDocument) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(doc.doc_id)
        payload = {
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "doc": doc.model_dump(mode="json"),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[cache] SAVED {doc.doc_id} -> {path}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Strategy helpers
    # ------------------------------------------------------------------

    def _initial_strategy(self, profile: DocumentProfile) -> str:
        cost = profile.estimated_extraction_cost
        if cost == "fast_text_sufficient":
            return "fast_text"
        if cost == "needs_vision_model":
            return "vision"
        return "layout"

    def _get_extractor(self, strategy: str):
        return {"fast_text": self._fast, "layout": self._layout, "vision": self._vision}[strategy]

    def _extract_with(
        self,
        strategy: str,
        doc_path: Path,
        profile: DocumentProfile,
        page_numbers: set[int] | None = None,
    ) -> tuple[ExtractedDocument, float]:
        t0 = time.perf_counter()
        doc = self._get_extractor(strategy).extract(doc_path, profile, page_numbers=page_numbers)
        elapsed = time.perf_counter() - t0
        return doc, elapsed

    def _cost_estimate(self, strategy: str, page_count: int) -> float:
        if strategy == "fast_text":
            return 0.0
        if strategy == "vision":
            return min(self.max_cost_per_doc, page_count * self.cost_per_image_usd)
        return page_count * self.cost_per_layout_page_usd

    def _budget_exceeded(self, strategy: str, page_count: int) -> bool:
        if strategy == "vision":
            return page_count * self.cost_per_image_usd > self.max_cost_per_doc
        if strategy == "layout":
            return page_count * self.cost_per_layout_page_usd > self.max_cost_per_doc
        return False

    def _append_ledger(
        self,
        doc_id: str,
        strategy_used: str,
        confidence_score: float,
        cost_estimate_usd: float,
        processing_time_sec: float,
        escalated_from: str | None,
        error: str | None = None,
    ) -> None:
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "doc_id": doc_id,
            "strategy_used": strategy_used,
            "confidence_score": confidence_score,
            "cost_estimate_usd": round(cost_estimate_usd, 6),
            "processing_time_sec": round(processing_time_sec, 3),
            "escalated_from": escalated_from,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if error is not None:
            entry["error"] = error
        with open(self._ledger_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------
    # Document-level extraction (fallback when baseline == fast_text)
    # ------------------------------------------------------------------

    def _extract_document_level(
        self,
        path: Path,
        profile: DocumentProfile,
    ) -> ExtractedDocument:
        """Original document-level routing with escalation/fallback chain."""
        strategy = self._initial_strategy(profile)
        escalated_from: str | None = None
        esc = self._rules.get("escalation", {})
        use_fallback = esc.get("fallback_on_failure", True)
        fallback_chain = {"vision": "layout", "layout": "fast_text", "fast_text": None} if use_fallback else {}

        while True:
            if self._budget_exceeded(strategy, profile.page_count):
                if self._on_budget_exceeded == "use_strategy_b" and strategy == "vision":
                    print(
                        f"[budget] vision cost exceeds cap, downgrading to layout",
                        file=sys.stderr,
                    )
                    escalated_from = escalated_from or strategy
                    strategy = "layout"

            try:
                result, elapsed = self._extract_with(strategy, path, profile)
            except Exception as e:
                cost = self._cost_estimate(strategy, profile.page_count)
                self._append_ledger(
                    doc_id=profile.doc_id, strategy_used=strategy,
                    confidence_score=0.0, cost_estimate_usd=cost,
                    processing_time_sec=0.0, escalated_from=escalated_from,
                    error=str(e),
                )
                next_strategy = fallback_chain.get(strategy)
                if next_strategy:
                    escalated_from = strategy
                    strategy = next_strategy
                    continue
                raise

            cost = self._cost_estimate(strategy, profile.page_count)
            self._append_ledger(
                doc_id=profile.doc_id, strategy_used=strategy,
                confidence_score=result.confidence_score,
                cost_estimate_usd=cost, processing_time_sec=elapsed,
                escalated_from=escalated_from,
            )

            if strategy == "fast_text" and result.confidence_score < self.strategy_a_min:
                escalated_from = "fast_text"
                strategy = "layout"
                continue
            if strategy == "layout" and result.confidence_score < self.strategy_b_min:
                escalated_from = "layout"
                strategy = "vision"
                continue
            return result

    # ------------------------------------------------------------------
    # Per-page extraction with mixed-strategy support
    # ------------------------------------------------------------------

    def _extract_per_page(
        self,
        path: Path,
        profile: DocumentProfile,
        baseline: str,
    ) -> ExtractedDocument:
        """Page-level routing: use PageAnalyzer to classify each page, run
        fast_text on suitable pages, baseline on the rest, and escalate
        individual pages whose per-page confidence is too low.
        """
        page_decisions = self._page_analyzer.classify_pages(
            profile.doc_id, baseline, profile.page_count,
        )

        fast_pages = {pg for pg, s in page_decisions.items() if s == "fast_text"}
        baseline_pages = {pg for pg, s in page_decisions.items() if s != "fast_text"}

        partial_docs: list[ExtractedDocument] = []

        # --- Phase 1: fast_text on downgraded pages ---
        if fast_pages:
            try:
                ft_result, ft_elapsed = self._extract_with("fast_text", path, profile, page_numbers=fast_pages)
            except Exception:
                baseline_pages |= fast_pages
                fast_pages = set()
                ft_result = None
                ft_elapsed = 0.0

            if ft_result is not None:
                # Per-page confidence check: escalate pages that fell below threshold
                escalated_pages: set[int] = set()
                kept_pages: list[ExtractedPage] = []
                for pg in ft_result.pages:
                    if pg.confidence is not None and pg.confidence < self.strategy_a_min:
                        escalated_pages.add(pg.page_number)
                    else:
                        kept_pages.append(pg)

                if escalated_pages:
                    print(
                        f"[router] {len(escalated_pages)} fast_text pages below "
                        f"confidence {self.strategy_a_min}, escalating to {baseline}",
                        file=sys.stderr,
                    )
                    baseline_pages |= escalated_pages

                if kept_pages:
                    kept_doc = ExtractedDocument(
                        doc_id=profile.doc_id, profile=profile,
                        pages=kept_pages,
                        tables=[t for t in ft_result.tables if t.page_number not in escalated_pages],
                        figures=[], strategy_used="fast_text",
                        confidence_score=ft_result.confidence_score,
                    )
                    partial_docs.append(kept_doc)

                cost = self._cost_estimate("fast_text", len(fast_pages))
                self._append_ledger(
                    doc_id=profile.doc_id, strategy_used="fast_text",
                    confidence_score=ft_result.confidence_score,
                    cost_estimate_usd=cost, processing_time_sec=ft_elapsed,
                    escalated_from=None,
                )

        # --- Phase 2: baseline strategy on remaining pages ---
        if baseline_pages:
            effective_baseline = baseline
            if self._budget_exceeded(effective_baseline, len(baseline_pages)):
                if self._on_budget_exceeded == "use_strategy_b" and effective_baseline == "vision":
                    effective_baseline = "layout"

            esc_cfg = self._rules.get("escalation", {})
            use_fallback = esc_cfg.get("fallback_on_failure", True)
            fallback_chain = {"vision": "layout", "layout": "fast_text", "fast_text": None} if use_fallback else {}
            escalated_from: str | None = None
            strategy = effective_baseline

            while True:
                try:
                    bl_result, bl_elapsed = self._extract_with(strategy, path, profile, page_numbers=baseline_pages)
                except Exception as e:
                    cost = self._cost_estimate(strategy, len(baseline_pages))
                    self._append_ledger(
                        doc_id=profile.doc_id, strategy_used=strategy,
                        confidence_score=0.0, cost_estimate_usd=cost,
                        processing_time_sec=0.0, escalated_from=escalated_from,
                        error=str(e),
                    )
                    next_strategy = fallback_chain.get(strategy)
                    if next_strategy:
                        escalated_from = strategy
                        strategy = next_strategy
                        continue
                    raise

                cost = self._cost_estimate(strategy, len(baseline_pages))
                self._append_ledger(
                    doc_id=profile.doc_id, strategy_used=strategy,
                    confidence_score=bl_result.confidence_score,
                    cost_estimate_usd=cost, processing_time_sec=bl_elapsed,
                    escalated_from=escalated_from,
                )

                if strategy == "layout" and bl_result.confidence_score < self.strategy_b_min:
                    escalated_from = "layout"
                    strategy = "vision"
                    continue

                partial_docs.append(bl_result)
                break

        if not partial_docs:
            return self._extract_document_level(path, profile)

        return _merge_extracted_documents(profile.doc_id, profile, partial_docs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        doc_path: Path | str,
        profile: DocumentProfile,
        force_reextract: bool = False,
    ) -> ExtractedDocument:
        """Run extraction with per-page routing and escalation.

        1. Load DocumentProfile → determine baseline strategy.
        2. If baseline is fast_text → run document-level extraction (no benefit
           from page analysis when the cheapest strategy is already selected).
        3. Otherwise → use PageAnalyzer to classify each page, run fast_text
           on suitable pages, baseline on the rest, escalate low-confidence pages.
        4. Merge results into a single ExtractedDocument.
        5. Cache the final result and log to the extraction ledger.
        """
        path = Path(doc_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        if not force_reextract:
            cached = self._load_from_cache(profile.doc_id)
            if cached is not None:
                return cached

        baseline = self._initial_strategy(profile)

        if baseline == "fast_text":
            result = self._extract_document_level(path, profile)
        else:
            result = self._extract_per_page(path, profile, baseline)

        self._save_to_cache(result)
        return result


from ledger.agents.refinery.utils.doc_id import stem_to_doc_id as _stem_to_doc_id


def main() -> None:
    """CLI: python -m src.agents.extractor [--force] <path_to_pdf> [path_to_pdf ...]

    --force   Bypass the extraction cache and re-run extraction even if a cached
              result exists at .refinery/extractions/{doc_id}.json.

    Loads profile from .refinery/profiles/ or runs triage first if missing.
    Saves ExtractedDocument to .refinery/extractions/{doc_id}.json after each run.
    """
    import os

    def _ts() -> str:
        return datetime.now(timezone.utc).isoformat()

    argv = sys.argv[1:]
    force_reextract = "--force" in argv
    argv = [a for a in argv if a != "--force"]

    if not argv:
        print(
            "Usage: python -m src.agents.extractor [--force] <path_to_pdf> [path_to_pdf ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[{_ts()}] extractor main started, force={force_reextract}, argv={argv}", file=sys.stderr)

    base = Path(__file__).resolve().parent.parent.parent
    profiles_dir = base / ".refinery" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    from src.agents.triage import TriageAgent

    stats_dir = base / ".refinery" / "output" / "output_pdfplumber"
    triage = TriageAgent()
    router = ExtractionRouter(
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        stats_dir=stats_dir,
    )
    print(f"[{_ts()}] ledger path: {router._ledger_path}", file=sys.stderr)
    print(f"[{_ts()}] cache dir:   {router._cache_dir}", file=sys.stderr)

    for arg in argv:
        path = Path(arg)
        if not path.is_absolute():
            path = base / path
        print(f"[{_ts()}] processing arg: {arg!r} -> resolved path: {path}", file=sys.stderr)
        if not path.exists():
            print(f"[{_ts()}] SKIP (not found): {path}", file=sys.stderr)
            continue
        doc_id = _stem_to_doc_id(path.stem)
        profile_file = profiles_dir / f"{doc_id}.json"
        if profile_file.exists():
            profile = DocumentProfile.model_validate_json(profile_file.read_text())
        else:
            profile = triage.profile_and_save(path, output_dir=profiles_dir)
        print(f"[{_ts()}] extraction about to start: path={path}, doc_id={profile.doc_id}", file=sys.stderr)
        try:
            result = router.extract(path, profile, force_reextract=force_reextract)
            print(f"[{_ts()}] extraction finished: {path.name} -> {result.strategy_used} conf={result.confidence_score:.2f}")
        except Exception as e:
            print(f"[{_ts()}] extraction failed: {path.name}: {e}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
