"""Layout-aware extraction via Docling — tables, figures, reading order."""

from pathlib import Path

from ledger.agents.refinery.models.document_profile import DocumentProfile
from ledger.agents.refinery.models.extracted_document import (
    BoundingBox,
    ExtractedDocument,
    ExtractedPage,
    ExtractedTable,
    ExtractedFigure,
)
from ledger.agents.refinery.strategies.base import BaseExtractor


def _load_docling():
    """Lazy import to avoid hard dependency at import time."""
    try:
        from docling.document_converter import DocumentConverter
        return DocumentConverter
    except ImportError:
        return None


def _create_cpu_converter():
    """Create DocumentConverter configured for CPU-only (no CUDA/NVIDIA)."""
    try:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.CPU,
        )
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
    except ImportError:
        return None


def _cell_to_value(cell) -> str | float | int:
    """Normalize a Docling table cell (TableCell or list of TableCell) to str, float, or int."""
    if cell is None:
        return ""
    if isinstance(cell, (str, int, float)):
        return cell
    # Docling: cell can be a list of TableCell (e.g. merged cells)
    if isinstance(cell, (list, tuple)):
        if not cell:
            return ""
        cell = cell[0]
    # TableCell-like: expect .text or .value
    text = getattr(cell, "text", None) or getattr(cell, "value", None)
    if text is None:
        return str(cell) if cell else ""
    s = str(text).strip()
    if not s:
        return ""
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s.replace(",", ""))
    except ValueError:
        pass
    return s


def _normalize_table_rows(data) -> list[list[str | float | int]]:
    """Convert Docling table data (rows of cells) to list[list[str|float|int]]."""
    if not data:
        return []
    out: list[list[str | float | int]] = []
    for row in data:
        if not isinstance(row, (list, tuple)):
            continue
        out.append([_cell_to_value(c) for c in row])
    return out


def _prov_page(item) -> int:
    """Return the page number from a Docling v2 item's prov list (prov[0].page_no)."""
    prov = getattr(item, "prov", None)
    if prov:
        try:
            return int(prov[0].page_no)
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
    return getattr(item, "page_no", 1) or 1


def _prov_bbox(item) -> BoundingBox | None:
    """Return a BoundingBox from a Docling v2 item's prov list (prov[0].bbox)."""
    prov = getattr(item, "prov", None)
    if prov:
        try:
            b = prov[0].bbox
            if b is None:
                return None
            return BoundingBox(
                x0=float(getattr(b, "l", 0)),
                y0=float(getattr(b, "t", 0)),
                x1=float(getattr(b, "r", 0)),
                y1=float(getattr(b, "b", 0)),
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
    return None


def _table_grid_to_rows(t) -> tuple[list[str], list[list[str | float | int]]]:
    """Extract (headers, rows) from a Docling v2 TableItem using data.grid."""
    data = getattr(t, "data", None)
    grid = getattr(data, "grid", None) if data is not None else None
    if grid is None:
        # Legacy flat list fallback
        raw = getattr(t, "data", []) or []
        return [], _normalize_table_rows(raw)

    headers: list[str] = []
    rows: list[list[str | float | int]] = []
    for row_cells in grid:
        row_vals = [_cell_to_value(c) for c in row_cells]
        # Mark as header if all cells in this row are column_header
        if all(getattr(c, "column_header", False) for c in row_cells if c is not None):
            headers = [str(v) for v in row_vals]
        else:
            rows.append(row_vals)
    return headers, rows


def _page_confidence(page: ExtractedPage) -> float:
    """Score a single page's extraction quality based on text yield and structure."""
    text_len = len((page.text or "").strip())
    has_blocks = bool(page.blocks)

    if text_len >= 500:
        text_score = 1.0
    elif text_len >= 200:
        text_score = 0.7 + 0.3 * (text_len - 200) / 300
    elif text_len >= 50:
        text_score = 0.4 + 0.3 * (text_len - 50) / 150
    elif text_len > 0:
        text_score = 0.15 + 0.25 * (text_len / 50)
    else:
        text_score = 0.0

    structure_bonus = 0.1 if has_blocks else 0.0
    return round(min(1.0, text_score + structure_bonus), 4)


def _compute_confidence(
    pages: list[ExtractedPage],
    tables: list[ExtractedTable],
    figures: list[ExtractedFigure],
    profile: DocumentProfile,
) -> float:
    """Quality-derived confidence from per-page text yield, table completeness,
    and structural richness.  Returns a continuous 0–1 score instead of
    hard-coded tiers, enabling the escalation guard to react proportionally.
    """
    if not pages:
        return 0.0

    total_pages = len(pages)

    # --- Signal 1: aggregate per-page text yield (weight 0.40) ---
    page_scores = [_page_confidence(p) for p in pages]
    avg_text_yield = sum(page_scores) / total_pages

    # --- Signal 2: empty-page penalty (weight 0.25) ---
    empty_pages = sum(1 for p in pages if len((p.text or "").strip()) < 10)
    empty_penalty = 1.0 - (empty_pages / total_pages)

    # --- Signal 3: table completeness (weight 0.20) ---
    if tables:
        complete = sum(1 for t in tables if t.headers and t.rows)
        table_score = complete / len(tables)
    else:
        expects_tables = profile.layout_complexity in (
            "table_heavy", "multi_column", "figure_heavy", "mixed",
        )
        table_score = 0.5 if expects_tables else 0.8

    # --- Signal 4: structural richness (weight 0.15) ---
    has_tables = bool(tables)
    has_figures = bool(figures)
    has_blocks = any(p.blocks for p in pages)
    structure_count = sum([has_tables, has_figures, has_blocks])
    structure_score = min(1.0, 0.4 + structure_count * 0.2)

    confidence = (
        avg_text_yield * 0.40
        + empty_penalty * 0.25
        + table_score * 0.20
        + structure_score * 0.15
    )
    return round(max(0.0, min(1.0, confidence)), 4)


class DoclingDocumentAdapter:
    """Adapt Docling ConversionResult to our ExtractedDocument schema."""

    @staticmethod
    def convert(
        doc_path: Path | str,
        profile: DocumentProfile,
        conversion_result,
        page_numbers: set[int] | None = None,
    ) -> ExtractedDocument:
        """Build ExtractedDocument from Docling v2 pipeline result.

        Iterates doc.texts per-page to produce one ExtractedPage per PDF page.
        Tables and figures carry their correct page number from prov[0].page_no.
        If *page_numbers* is given, only those pages are included in the result.
        """
        doc = getattr(conversion_result, "document", conversion_result)

        # --- Build per-page text blocks from doc.texts ---
        page_texts: dict[int, list[str]] = {}
        for item in getattr(doc, "texts", []):
            pg = _prov_page(item)
            text = getattr(item, "text", None) or ""
            page_texts.setdefault(pg, []).append(text)

        # Fallback: if doc.texts is empty, export full markdown as page 1
        if not page_texts:
            export_md = getattr(doc, "export_to_markdown", None)
            full_md = export_md() if export_md else ""
            page_texts[1] = [full_md] if full_md else [""]

        pages: list[ExtractedPage] = []
        for pg, blocks in sorted(page_texts.items()):
            ep = ExtractedPage(
                page_number=pg, text="\n".join(blocks), blocks=[],
                strategy_used="layout",
            )
            ep.confidence = _page_confidence(ep)
            pages.append(ep)

        # --- Tables (Docling v2: use data.grid + prov) ---
        tables: list[ExtractedTable] = []
        for t in getattr(doc, "tables", []):
            pg = _prov_page(t)
            bbox = _prov_bbox(t)
            headers, rows = _table_grid_to_rows(t)
            tables.append(ExtractedTable(
                page_number=pg,
                bbox=bbox,
                headers=headers,
                rows=rows,
            ))

        # --- Figures / pictures ---
        figures: list[ExtractedFigure] = []
        for fig in getattr(doc, "pictures", []) or getattr(doc, "figures", []):
            pg = _prov_page(fig)
            bbox = _prov_bbox(fig)
            caption_item = getattr(fig, "caption", None)
            caption_text = (
                getattr(caption_item, "text", str(caption_item))
                if caption_item is not None else None
            )
            figures.append(ExtractedFigure(page_number=pg, bbox=bbox, caption=caption_text))

        if page_numbers is not None:
            pages = [p for p in pages if p.page_number in page_numbers]
            tables = [t for t in tables if t.page_number in page_numbers]
            figures = [f for f in figures if f.page_number in page_numbers]

        confidence_score = _compute_confidence(pages, tables, figures, profile)

        return ExtractedDocument(
            doc_id=profile.doc_id,
            profile=profile,
            pages=pages,
            tables=tables,
            figures=figures,
            strategy_used="layout",
            confidence_score=confidence_score,
        )


class LayoutExtractor(BaseExtractor):
    """Extract using Docling for layout-aware parsing. Uses CPU by default."""

    def __init__(self, use_cpu: bool = True):
        self._converter_class = _load_docling()
        self._use_cpu = use_cpu

    def extract(
        self,
        doc_path: Path | str,
        profile: DocumentProfile,
        page_numbers: set[int] | None = None,
    ) -> ExtractedDocument:
        path = Path(doc_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        if self._converter_class is None:
            return ExtractedDocument(
                doc_id=profile.doc_id,
                profile=profile,
                pages=[ExtractedPage(page_number=1, text="", blocks=[], strategy_used="layout")],
                tables=[],
                figures=[],
                strategy_used="layout",
                confidence_score=0.0,
            )

        if self._use_cpu:
            converter = _create_cpu_converter()
            if converter is None:
                converter = self._converter_class()
        else:
            converter = self._converter_class()

        result = converter.convert(str(path))
        return DoclingDocumentAdapter.convert(doc_path, profile, result, page_numbers=page_numbers)
