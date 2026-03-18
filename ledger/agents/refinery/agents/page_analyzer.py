"""Page-level analysis using precomputed pdfplumber statistics.

Loads character density JSON files produced by scripts/character_density.py
and classifies each page as suitable for fast_text (Strategy A) or requiring
the document's baseline strategy (B/C).
"""

import json
import re
import sys
from pathlib import Path

import yaml


def _load_rules(rules_path: Path | None = None) -> dict:
    if rules_path and rules_path.exists():
        with open(rules_path, "r") as f:
            return yaml.safe_load(f) or {}
    base = Path(__file__).resolve().parent.parent.parent
    for candidate in [base / "rubric" / "extraction_rules.yaml", base / "extraction_rules.yaml"]:
        if candidate.exists():
            with open(candidate, "r") as f:
                return yaml.safe_load(f) or {}
    return {}


def _normalize_key(name: str) -> str:
    """Collapse hyphens, dots, and special chars into underscores for fuzzy matching."""
    s = name.lower().replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^\w]", "", s)
    return re.sub(r"_+", "_", s)


class PageAnalyzer:
    """Classify individual pages using precomputed pdfplumber statistics.

    Signals used per page:
      - char_count vs min_chars_per_page
      - density_chars_per_pt2 vs char_density_threshold
      - whitespace_ratio (high whitespace → likely image/cover page)
      - text_coverage_ratio from bbox_distribution

    Pages that pass all thresholds are deemed suitable for Strategy A (fast_text).
    Pages that fail are routed to the document's baseline strategy.
    """

    def __init__(
        self,
        stats_dir: Path | str | None = None,
        rules_path: Path | None = None,
    ):
        _base = Path(__file__).resolve().parent.parent.parent
        self._stats_dir = Path(stats_dir) if stats_dir else _base / ".refinery" / "output" / "output_pdfplumber"
        rules = _load_rules(rules_path)
        triage = rules.get("triage", {})
        self._density_threshold = float(triage.get("char_density_threshold", 0.0005))
        self._min_chars = int(triage.get("min_chars_per_page", 100))
        self._max_whitespace = 0.85

        self._index: dict[str, Path] | None = None

    def _build_index(self) -> dict[str, Path]:
        """Build a normalized-key → file-path map for all density JSON files."""
        idx: dict[str, Path] = {}
        if not self._stats_dir.is_dir():
            return idx
        for f in self._stats_dir.glob("*_character_density.json"):
            stem = f.stem.replace("_character_density", "")
            idx[_normalize_key(stem)] = f
        return idx

    def _find_stats_file(self, doc_id: str) -> Path | None:
        if self._index is None:
            self._index = self._build_index()
        key = _normalize_key(doc_id)
        return self._index.get(key)

    def load_page_stats(self, doc_id: str) -> dict[int, dict] | None:
        """Load precomputed page stats. Returns ``{page_number: stats_dict}`` or None."""
        path = self._find_stats_file(doc_id)
        if path is None:
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {p["page"]: p for p in data.get("pages", [])}
        except Exception as exc:
            print(f"[page_analyzer] WARN failed to load {path}: {exc}", file=sys.stderr)
            return None

    def classify_pages(
        self,
        doc_id: str,
        baseline_strategy: str,
        total_pages: int,
    ) -> dict[int, str]:
        """Return ``{page_number: strategy_name}`` for every page in the document.

        Pages with strong text signals get ``"fast_text"``; the rest get *baseline_strategy*.
        If no precomputed stats are found, all pages receive *baseline_strategy*.
        """
        stats = self.load_page_stats(doc_id)
        if stats is None:
            print(
                f"[page_analyzer] no precomputed stats for {doc_id!r}, "
                f"using baseline {baseline_strategy!r} for all pages",
                file=sys.stderr,
            )
            return {pg: baseline_strategy for pg in range(1, total_pages + 1)}

        decisions: dict[int, str] = {}
        for pg in range(1, total_pages + 1):
            page_stats = stats.get(pg)
            if page_stats is None:
                decisions[pg] = baseline_strategy
                continue

            char_count = page_stats.get("char_count", 0)
            density = page_stats.get("density_chars_per_pt2", 0.0)
            whitespace = page_stats.get("whitespace_ratio", 0.0)
            bbox = page_stats.get("bbox_distribution") or {}
            coverage = bbox.get("text_coverage_ratio", 0.0)

            is_text_rich = (
                char_count >= self._min_chars
                and density >= self._density_threshold
                and whitespace < self._max_whitespace
                and coverage > 0.02
            )

            decisions[pg] = "fast_text" if is_text_rich else baseline_strategy

        fast_count = sum(1 for s in decisions.values() if s == "fast_text")
        print(
            f"[page_analyzer] {doc_id}: {fast_count}/{total_pages} pages → fast_text, "
            f"{total_pages - fast_count} → {baseline_strategy}",
            file=sys.stderr,
        )
        return decisions
