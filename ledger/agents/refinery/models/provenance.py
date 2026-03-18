"""Provenance chain — source citations for every answer."""

from pydantic import BaseModel, Field

from ledger.agents.refinery.models.extracted_document import BoundingBox


class SourceCitation(BaseModel):
    """One cited source (document, page, region)."""

    document_name: str = Field(..., description="Document filename or id")
    page_number: int = Field(..., ge=1)
    bounding_box: BoundingBox | None = Field(None)
    content_hash: str = Field(..., description="LDU content hash for verification")
    excerpt: str | None = Field(None, description="Short snippet of the source text")


class ProvenanceChain(BaseModel):
    """Answer plus list of source citations for audit."""

    answer: str = Field(..., description="Generated answer text")
    sources: list[SourceCitation] = Field(default_factory=list)
