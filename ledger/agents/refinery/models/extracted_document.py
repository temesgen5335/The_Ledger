"""Normalized extraction output — shared schema for all extraction strategies."""

from typing import Any

from pydantic import BaseModel, Field

from ledger.agents.refinery.models.document_profile import DocumentProfile


class BoundingBox(BaseModel):
    """Spatial coordinates (e.g. from pdfplumber or Docling)."""

    x0: float = Field(..., description="Left")
    y0: float = Field(..., description="Top")
    x1: float = Field(..., description="Right")
    y1: float = Field(..., description="Bottom")
    page: int | None = Field(None, description="Page number if known")


class ExtractedPage(BaseModel):
    """One page of extracted content."""

    page_number: int = Field(..., ge=1)
    text: str = Field(default="", description="Full text of the page")
    blocks: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Text blocks with bbox and reading order",
    )
    strategy_used: str | None = Field(
        None, description="Strategy that produced this page (enables mixed-strategy docs)"
    )
    confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Per-page extraction confidence"
    )


class ExtractedTable(BaseModel):
    """Structured table (headers + rows)."""

    page_number: int = Field(..., ge=1)
    bbox: BoundingBox | None = Field(None)
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str | float | int]] = Field(default_factory=list)
    raw_data: dict[str, Any] | None = Field(None, description="Original structure if needed")


class ExtractedFigure(BaseModel):
    """Figure with optional caption."""

    page_number: int = Field(..., ge=1)
    bbox: BoundingBox | None = Field(None)
    caption: str | None = Field(None)
    alt_text: str | None = Field(None)


class ExtractedDocument(BaseModel):
    """Unified extraction output from any strategy (A, B, or C)."""

    doc_id: str = Field(..., description="Document identifier")
    profile: DocumentProfile = Field(..., description="Triage profile used for extraction")
    pages: list[ExtractedPage] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    figures: list[ExtractedFigure] = Field(default_factory=list)
    strategy_used: str = Field(..., description="fast_text | layout | vision")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
