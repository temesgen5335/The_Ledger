"""PageIndex — hierarchical navigation structure over a document."""

from typing import Literal

from pydantic import BaseModel, Field

DataType = Literal["tables", "figures", "equations"]


class Section(BaseModel):
    """One node in the document table-of-contents tree."""

    title: str = Field(..., description="Section heading text")
    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)
    child_sections: list["Section"] = Field(default_factory=list)
    key_entities: list[str] = Field(
        default_factory=list,
        description="Extracted named entities (orgs, dates, amounts)",
    )
    summary: str | None = Field(None, description="2–3 sentence LLM-generated summary")
    data_types_present: list[DataType] = Field(default_factory=list)


class PageIndex(BaseModel):
    """Root of the document navigation tree."""

    doc_id: str = Field(..., description="Document identifier")
    root_section: Section = Field(..., description="Top-level section tree")
