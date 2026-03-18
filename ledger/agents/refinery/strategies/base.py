"""Abstract base for all extraction strategies."""

from abc import ABC, abstractmethod
from pathlib import Path

from ledger.agents.refinery.models.document_profile import DocumentProfile
from ledger.agents.refinery.models.extracted_document import ExtractedDocument


class BaseExtractor(ABC):
    """Contract for extraction strategies; all return the same ExtractedDocument schema."""

    @abstractmethod
    def extract(
        self,
        doc_path: Path | str,
        profile: DocumentProfile,
        page_numbers: set[int] | None = None,
    ) -> ExtractedDocument:
        """Extract structured content from the document.

        If *page_numbers* is provided, only those pages are processed (enables
        per-page routing where different strategies handle different pages).
        """
        ...
