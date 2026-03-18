"""Triage agent — classifies documents and produces DocumentProfile for extraction routing."""

from pathlib import Path

import pdfplumber
import yaml

from ledger.agents.refinery.models.document_profile import (
    DocumentProfile,
    DomainHint,
    EstimatedExtractionCost,
    LanguageDetection,
    LayoutComplexity,
    OriginType,
)


def _load_rules(rules_path: Path | None = None) -> dict:
    """Load extraction_rules.yaml from rubric or project root."""
    if rules_path and rules_path.exists():
        with open(rules_path, "r") as f:
            return yaml.safe_load(f) or {}
    base = Path(__file__).resolve().parent.parent.parent
    for candidate in [base / "rubric" / "extraction_rules.yaml", base / "extraction_rules.yaml"]:
        if candidate.exists():
            with open(candidate, "r") as f:
                return yaml.safe_load(f) or {}
    return {}


def _detect_origin_type(
    char_count_per_page: list[int],
    image_area_ratio: float,
    min_chars_per_page: int,
    image_area_ratio_threshold: float,
) -> OriginType:
    """Classify native_digital vs scanned_image vs mixed from character and image metrics."""
    if not char_count_per_page:
        return "scanned_image"
    low_char_pages = sum(1 for c in char_count_per_page if c < min_chars_per_page)
    pct_low = low_char_pages / len(char_count_per_page)
    if image_area_ratio >= image_area_ratio_threshold and pct_low > 0.5:
        return "scanned_image"
    if image_area_ratio > 0.2 and pct_low > 0.2:
        return "mixed"
    if pct_low > 0.5:
        return "scanned_image"
    return "native_digital"


def _detect_layout_complexity(
    page_count: int,
    has_tables: bool,
    table_count: int,
    char_count_total: int,
) -> LayoutComplexity:
    """Heuristic layout complexity from table presence and text volume."""
    if page_count == 0:
        return "single_column"
    if table_count >= page_count * 0.5 or (has_tables and table_count >= 3):
        return "table_heavy"
    if table_count >= 1 or char_count_total > 10000:
        return "multi_column"
    return "single_column"


def _detect_domain_hint(text_sample: str, keywords: dict[str, list[str]]) -> DomainHint:
    """Keyword-based domain hint from first pages' text."""
    sample = (text_sample or "").lower()
    scores: dict[str, int] = {}
    for domain, words in (keywords or {}).items():
        if domain == "general":
            continue
        scores[domain] = sum(1 for w in words if w.lower() in sample)
    if not scores:
        return "general"
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def _estimate_extraction_cost(origin_type: OriginType, layout_complexity: LayoutComplexity) -> EstimatedExtractionCost:
    """Map profile to recommended strategy tier."""
    if origin_type == "scanned_image":
        return "needs_vision_model"
    if origin_type == "mixed":
        return "needs_layout_model"
    if layout_complexity in ("multi_column", "table_heavy", "figure_heavy", "mixed"):
        return "needs_layout_model"
    return "fast_text_sufficient"


class TriageAgent:
    """Classifies documents and persists DocumentProfile for the extraction router."""

    def __init__(
        self,
        rules_path: Path | None = None,
        min_chars_per_page: int = 100,
        image_area_ratio_threshold: float = 0.5,
    ):
        self._rules = _load_rules(rules_path)
        triage = self._rules.get("triage", {})
        self.min_chars_per_page = min_chars_per_page or triage.get("min_chars_per_page", 100)
        self.image_area_ratio_threshold = image_area_ratio_threshold or triage.get(
            "image_area_ratio_threshold", 0.5
        )
        self._domain_keywords = self._rules.get("domain_hint_keywords", {})

    def profile(self, doc_path: Path | str) -> DocumentProfile:
        """Analyze document and return a DocumentProfile. Does not persist."""
        path = Path(doc_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        from src.utils.doc_id import stem_to_doc_id
        doc_id = stem_to_doc_id(path.stem)

        char_count_per_page: list[int] = []
        total_char_count = 0
        total_page_area = 0.0
        total_image_area = 0.0
        table_count = 0
        text_samples: list[str] = []

        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                w = page.width or 0
                h = page.height or 0
                area = float(w * h) if w and h else 0.0
                total_page_area += area

                text = page.extract_text() or ""
                chars = len(text.replace(" ", ""))
                char_count_per_page.append(chars)
                total_char_count += chars
                if i < 3:
                    text_samples.append(text)

                # Approximate image area
                img_area = 0.0
                for im in getattr(page, "images", []) or []:
                    img_area += (im.get("width") or 0) * (im.get("height") or 0)
                total_image_area += img_area

                tables = page.extract_tables() or []
                table_count += len([t for t in tables if t])

            image_ratio = total_image_area / total_page_area if total_page_area > 0 else 0.0

        origin_type = _detect_origin_type(
            char_count_per_page,
            image_ratio,
            self.min_chars_per_page,
            self.image_area_ratio_threshold,
        )
        layout_complexity = _detect_layout_complexity(
            page_count, table_count > 0, table_count, total_char_count
        )
        sample = " ".join(text_samples[:3])
        domain_hint = _detect_domain_hint(sample, self._domain_keywords)
        estimated_extraction_cost = _estimate_extraction_cost(origin_type, layout_complexity)

        # Simple language guess: assume English if we have enough Latin chars
        lang_code = "en"
        lang_confidence = 0.8
        if sample:
            alpha = sum(1 for c in sample if c.isalpha())
            if alpha > 50:
                lang_confidence = min(0.95, 0.7 + alpha / 1000)

        return DocumentProfile(
            doc_id=doc_id,
            origin_type=origin_type,
            layout_complexity=layout_complexity,
            language=LanguageDetection(code=lang_code, confidence=round(lang_confidence, 4)),
            domain_hint=domain_hint,
            estimated_extraction_cost=estimated_extraction_cost,
            page_count=page_count,
        )

    def profile_and_save(self, doc_path: Path | str, output_dir: Path | str | None = None) -> DocumentProfile:
        """Compute profile and persist to .refinery/profiles/{doc_id}.json."""
        profile = self.profile(doc_path)
        out_dir = Path(output_dir) if output_dir else Path.cwd() / ".refinery" / "profiles"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{profile.doc_id}.json"
        with open(out_file, "w") as f:
            f.write(profile.model_dump_json(indent=2))
        return profile


def main() -> None:
    """CLI entry: python -m src.agents.triage <path_to_pdf>."""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.agents.triage <path_to_pdf>", file=sys.stderr)
        sys.exit(1)
    path = Path(sys.argv[1])
    agent = TriageAgent()
    profile = agent.profile_and_save(path)
    print(profile.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
