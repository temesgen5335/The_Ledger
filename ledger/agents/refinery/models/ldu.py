"""Logical Document Unit — RAG-ready chunk with provenance metadata."""

from typing import Literal

from pydantic import BaseModel, Field

from ledger.agents.refinery.models.extracted_document import BoundingBox

ChunkType = Literal["paragraph", "table", "figure", "list", "heading"]


class LDU(BaseModel):
    """Semantically coherent chunk for retrieval; carries spatial and structural context."""

    ldu_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Chunk text")
    chunk_type: ChunkType = Field(..., description="Paragraph, table, figure, list, or heading")
    page_refs: list[int] = Field(default_factory=list, description="Page numbers this chunk spans")
    bounding_box: BoundingBox | None = Field(None)
    parent_section: str | None = Field(None, description="Section title from PageIndex")
    token_count: int = Field(..., ge=0)
    content_hash: str = Field(..., description="Deterministic hash for provenance verification")
