"""Fast text extraction via pdfplumber — low cost, confidence-gated."""

from pathlib import Path

import pdfplumber

from ledger.agents.refinery.models.document_profile import DocumentProfile
from ledger.agents.refinery.models.extracted_document import (
    BoundingBox,
    ExtractedDocument,
    ExtractedPage,
    ExtractedTable,
    ExtractedFigure,
)
from ledger.agents.refinery.strategies.base import BaseExtractor


def _compute_confidence(
    char_count: int,
    page_area: float,
    image_area_ratio: float,
    min_chars_per_page: int = 100,
) -> float:
    """Multi-signal confidence: char density and image ratio. Returns 0–1."""
    if page_area <= 0:
        return 0.0
    char_density = char_count / page_area if page_area else 0
    # Heuristic: good density > 0.001, low image ratio < 0.5
    density_score = min(1.0, char_density * 500)
    char_ok = 1.0 if char_count >= min_chars_per_page else char_count / min_chars_per_page
    image_penalty = 1.0 - image_area_ratio
    return max(0.0, min(1.0, (density_score * 0.3 + char_ok * 0.4 + image_penalty * 0.3)))


class FastTextExtractor(BaseExtractor):
    """Extract text with pdfplumber; score confidence for escalation guard."""

    def __init__(self, min_chars_per_page: int = 100):
        self.min_chars_per_page = min_chars_per_page

    def _extract_page(self, page, page_number: int) -> tuple[ExtractedPage, list[ExtractedTable], float, float, float]:
        """Extract a single pdfplumber page. Returns (page, tables, char_count, area, img_area)."""
        area = float(page.width * page.height) if page.width and page.height else 0.0
        text = page.extract_text() or ""
        char_count = len(text.replace(" ", ""))

        img_area = 0.0
        if hasattr(page, "images") and page.images:
            for im in page.images:
                w = im.get("width") or 0
                h = im.get("height") or 0
                img_area += w * h
        if hasattr(page, "curves") and page.curves:
            for c in page.curves:
                x0, x1 = c.get("x0", 0), c.get("x1", 0)
                y0, y1 = c.get("top", 0), c.get("bottom", 0)
                img_area += abs((x1 - x0) * (y1 - y0))

        blocks = []
        chars = getattr(page, "chars", None) or []
        for char in chars[:500]:
            blocks.append({
                "x0": char.get("x0"), "top": char.get("top"),
                "x1": char.get("x1"), "bottom": char.get("bottom"),
                "text": char.get("text", ""),
            })

        img_ratio = img_area / area if area > 0 else 0.0
        page_confidence = _compute_confidence(char_count, area, img_ratio, self.min_chars_per_page)

        extracted_page = ExtractedPage(
            page_number=page_number,
            text=text,
            blocks=blocks,
            strategy_used="fast_text",
            confidence=round(page_confidence, 4),
        )

        page_tables: list[ExtractedTable] = []
        raw_tables = page.extract_tables() or []
        for t in raw_tables:
            if not t:
                continue
            headers = t[0] if t else []
            rows = t[1:] if len(t) > 1 else []
            page_tables.append(
                ExtractedTable(
                    page_number=page_number,
                    headers=[str(h) for h in headers],
                    rows=[[cell for cell in row] for row in rows],
                )
            )

        return extracted_page, page_tables, float(char_count), area, img_area

    def extract(
        self,
        doc_path: Path | str,
        profile: DocumentProfile,
        page_numbers: set[int] | None = None,
    ) -> ExtractedDocument:
        """Extract text with pdfplumber. If *page_numbers* is given, only those pages are processed."""
        path = Path(doc_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        pages: list[ExtractedPage] = []
        tables: list[ExtractedTable] = []
        total_chars = 0.0
        total_area = 0.0
        total_image_area = 0.0

        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                if page_numbers is not None and i not in page_numbers:
                    continue
                ep, pt, cc, area, img_area = self._extract_page(page, i)
                pages.append(ep)
                tables.extend(pt)
                total_chars += cc
                total_area += area
                total_image_area += img_area

        image_ratio = total_image_area / total_area if total_area > 0 else 0.0
        confidence = _compute_confidence(
            int(total_chars), total_area, image_ratio, self.min_chars_per_page,
        )

        return ExtractedDocument(
            doc_id=profile.doc_id,
            profile=profile,
            pages=pages,
            tables=tables,
            figures=[],
            strategy_used="fast_text",
            confidence_score=round(confidence, 4),
        )
