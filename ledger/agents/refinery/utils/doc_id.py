"""Canonical doc_id derivation from PDF filename stems."""

import re


def stem_to_doc_id(stem: str) -> str:
    """Normalize a PDF filename stem into a stable document identifier.

    >>> stem_to_doc_id("CBE ANNUAL REPORT 2023-24")
    'cbe_annual_report_2023-24'
    """
    s = stem.replace(" ", "_").lower()
    return re.sub(r"[^\w\-]", "", s) or "document"
