"""Document profile produced by the triage agent — governs extraction strategy selection."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

OriginType = Literal["native_digital", "scanned_image", "mixed", "form_fillable"]
LayoutComplexity = Literal[
    "single_column", "multi_column", "table_heavy", "figure_heavy", "mixed"
]
DomainHint = Literal["financial", "legal", "technical", "medical", "general"]
EstimatedExtractionCost = Literal[
    "fast_text_sufficient", "needs_layout_model", "needs_vision_model"
]


class LanguageDetection(BaseModel):
    """Detected language with confidence."""

    code: str = Field(..., description="ISO 639-1 language code")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence 0–1")


class DocumentProfile(BaseModel):
    """Structured fingerprint of a document produced by the triage agent."""

    doc_id: str = Field(..., description="Unique document identifier")
    origin_type: OriginType = Field(..., description="Digital vs scanned vs mixed")
    layout_complexity: LayoutComplexity = Field(
        ..., description="Column layout and content type complexity"
    )
    language: LanguageDetection = Field(..., description="Detected language and confidence")
    domain_hint: DomainHint = Field(
        ..., description="Domain used to select extraction prompt strategy"
    )
    estimated_extraction_cost: EstimatedExtractionCost = Field(
        ..., description="Which strategy tier is recommended"
    )
    page_count: int = Field(..., ge=1, description="Number of pages")
    created_at: datetime = Field(default_factory=datetime.utcnow)
