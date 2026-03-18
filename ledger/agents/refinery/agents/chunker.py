"""Semantic Chunking Engine — ExtractedDocument to LDUs with five enforced rules."""

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from ledger.agents.refinery.models.extracted_document import (
    BoundingBox,
    ExtractedDocument,
    ExtractedFigure,
    ExtractedPage,
    ExtractedTable,
)
from ledger.agents.refinery.models.ldu import LDU, ChunkType
from ledger.agents.refinery.utils.tokenizer import count_tokens


def _load_rules(rules_path: Path | None = None) -> dict:
    base = Path(__file__).resolve().parent.parent.parent
    if rules_path and rules_path.exists():
        with open(rules_path, "r") as f:
            return yaml.safe_load(f) or {}
    for candidate in [base / "rubric" / "extraction_rules.yaml", base / "extraction_rules.yaml"]:
        if candidate.exists():
            with open(candidate, "r") as f:
                return yaml.safe_load(f) or {}
    return {}


def _content_hash(content: str, page_refs: list[int], bbox: BoundingBox | None) -> str:
    """Deterministic hash for provenance verification (spatial/content)."""
    parts = [content, repr(sorted(page_refs))]
    if bbox is not None:
        parts.append(f"{bbox.x0},{bbox.y0},{bbox.x1},{bbox.y1}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _extract_sections_from_markdown(text: str) -> list[tuple[int, str]]:
    """Return list of (level, title) for markdown headings (##, ###, etc.)."""
    out: list[tuple[int, str]] = []
    for m in re.finditer(r"^(#{1,6})\s+(.+)$", text, re.MULTILINE):
        level = len(m.group(1))
        title = m.group(2).strip()
        if title:
            out.append((level, title))
    return out


_PLAIN_HEADING_RE = re.compile(
    r"^(?:Part\s*\d+|Section\s*\d+|Chapter\s*\d+|\d+[\.\)]\s+[A-Z])[^$]*$",
    re.IGNORECASE,
)


def _looks_like_plain_heading(text: str) -> bool:
    """Heuristic: single short line that acts as a section header in plain text.

    Matches lines like 'Part 1: Rewritten update', 'Section 2', '3. Results'.
    Also catches any single short line ending with ':' that isn't a list item.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) != 1:
        return False
    line = lines[0]
    if len(line) > 120:
        return False
    if _PLAIN_HEADING_RE.match(line):
        return True
    # Short line ending with ':' that doesn't start with a bullet
    if line.endswith(":") and not re.match(r"^[●•\-\*]", line):
        return True
    return False


def _looks_like_list(text: str) -> bool:
    """Heuristic: multiple lines that look like numbered list items."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 2:
        return False
    numbered = sum(1 for l in lines if re.match(r"^\d+[\.\)]\s", l))
    return numbered >= 2 or numbered == len(lines)


def _split_list_by_items(text: str, max_tokens: int = 512) -> list[str]:
    """Split list text by item boundaries for list_unity max_tokens."""
    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for line in lines:
        line_tokens = count_tokens(line)
        if current_tokens + line_tokens > max_tokens and current:
            chunks.append("\n".join(current))
            current = [line]
            current_tokens = line_tokens
        else:
            current.append(line)
            current_tokens += line_tokens
    if current:
        chunks.append("\n".join(current))
    return chunks if chunks else [text]


def _split_by_heading_level(
    text: str, headings: list[tuple[int, str]]
) -> list[tuple[str | None, str]]:
    """Split text into (section_title_or_None, content) segments."""
    if not text.strip():
        return []
    if not headings:
        return [(None, text.strip())]
    segments: list[tuple[str | None, str]] = []
    current_section: str | None = None
    pos = 0
    for _, title in headings:
        pattern = r"#+\s*" + re.escape(title)
        match = re.search(pattern, text[pos:], re.IGNORECASE)
        if match:
            start = pos + match.start()
            end = pos + match.end()
            if start > pos:
                segments.append((current_section, text[pos:start].strip()))
            current_section = title
            pos = end
    if pos < len(text):
        segments.append((current_section, text[pos:].strip()))
    return segments if segments else [(None, text.strip())]


class ChunkValidator:
    """Verifies no chunking rule is violated before emitting an LDU."""

    def __init__(
        self,
        max_tokens_per_ldu: int = 512,
        table_integrity: bool = True,
        figure_cohesion: bool = True,
        list_unity: bool = True,
        section_inheritance: bool = True,
        cross_reference_resolution: bool = True,
    ):
        self.max_tokens = max_tokens_per_ldu
        self.table_integrity = table_integrity
        self.figure_cohesion = figure_cohesion
        self.list_unity = list_unity
        self.section_inheritance = section_inheritance
        self.cross_reference_resolution = cross_reference_resolution

    def validate(self, ldu: LDU) -> tuple[bool, str | None]:
        """Return (ok, error_message). If ok is True, error_message is None."""
        if ldu.chunk_type == "table" and self.table_integrity:
            if not ldu.content.strip():
                return False, "table_integrity: table LDU has no content"
            if "\n" not in ldu.content and "," not in ldu.content and "\t" not in ldu.content:
                return False, "table_integrity: table LDU appears to be a fragment"

        if ldu.chunk_type == "figure" and self.figure_cohesion:
            if not ldu.content.strip():
                return False, "figure_cohesion: figure LDU has no content"

        if ldu.chunk_type == "list" and self.list_unity:
            if ldu.token_count > self.max_tokens:
                return False, "list_unity: list LDU exceeds max_tokens_per_ldu"

        if self.cross_reference_resolution:
            if not ldu.content_hash:
                return False, "cross_reference_resolution: content_hash required for provenance"

        if ldu.token_count > self.max_tokens and ldu.chunk_type not in ("table", "figure"):
            return False, "chunk exceeds max_tokens_per_ldu"

        return True, None


class ChunkingEngine:
    """Converts ExtractedDocument into a list of LDUs."""

    def __init__(
        self,
        rules_path: Path | None = None,
        max_tokens_per_ldu: int | None = None,
    ):
        self._rules = _load_rules(rules_path)
        chunk_cfg = self._rules.get("chunking", {})
        self.max_tokens = max_tokens_per_ldu or chunk_cfg.get("max_tokens_per_ldu", 512)
        self.validator = ChunkValidator(
            max_tokens_per_ldu=self.max_tokens,
            table_integrity=chunk_cfg.get("table_integrity", True),
            figure_cohesion=chunk_cfg.get("figure_cohesion", True),
            list_unity=chunk_cfg.get("list_unity", True),
            section_inheritance=chunk_cfg.get("section_inheritance", True),
            cross_reference_resolution=chunk_cfg.get("cross_reference_resolution", True),
        )

    def chunk(self, doc: ExtractedDocument) -> list[LDU]:
        """Produce validated LDUs from ExtractedDocument."""
        doc_id = doc.doc_id
        ldus: list[LDU] = []
        section_stack: list[str] = []
        idx_by_type: dict[ChunkType, int] = {}

        for i, table in enumerate(doc.tables):
            rows_text = "\n".join(
                "\t".join(str(c) for c in row) for row in ([table.headers] + table.rows)
            )
            header_line = " | ".join(str(h) for h in table.headers) if table.headers else ""
            content = header_line + "\n" + rows_text
            if not content.strip():
                content = rows_text or "(empty table)"
            page_refs = [table.page_number]
            bbox = table.bbox
            token_count = count_tokens(content)
            content_hash = _content_hash(content, page_refs, bbox)
            ldu_id = f"{doc_id}_table_{i}"
            parent = section_stack[-1] if section_stack else None
            ldu = LDU(
                ldu_id=ldu_id,
                content=content,
                chunk_type="table",
                page_refs=page_refs,
                bounding_box=bbox,
                parent_section=parent,
                token_count=token_count,
                content_hash=content_hash,
            )
            ok, err = self.validator.validate(ldu)
            if ok:
                ldus.append(ldu)
            idx_by_type["table"] = idx_by_type.get("table", 0) + 1

        for i, fig in enumerate(doc.figures):
            content = fig.caption or fig.alt_text or "(figure)"
            if fig.caption and fig.alt_text:
                content = f"{fig.caption}\n{fig.alt_text}"
            page_refs = [fig.page_number]
            bbox = fig.bbox
            token_count = count_tokens(content)
            content_hash = _content_hash(content, page_refs, bbox)
            ldu_id = f"{doc_id}_figure_{i}"
            parent = section_stack[-1] if section_stack else None
            ldu = LDU(
                ldu_id=ldu_id,
                content=content,
                chunk_type="figure",
                page_refs=page_refs,
                bounding_box=bbox,
                parent_section=parent,
                token_count=token_count,
                content_hash=content_hash,
            )
            ok, err = self.validator.validate(ldu)
            if ok:
                ldus.append(ldu)
            idx_by_type["figure"] = idx_by_type.get("figure", 0) + 1

        is_single_markdown = (
            len(doc.pages) == 1
            and doc.pages[0].text
            and ("##" in doc.pages[0].text or "# " in doc.pages[0].text)
        )
        if is_single_markdown:
            full_text = doc.pages[0].text
            headings = _extract_sections_from_markdown(full_text)
            segments = _split_by_heading_level(full_text, headings)
            page_num = doc.pages[0].page_number
            for seg_title, seg_content in segments:
                if seg_title:
                    section_stack = [seg_title]
                    h_ldu = self._make_text_ldu(
                        doc_id, seg_title, "heading", [page_num], None, None, idx_by_type
                    )
                    if h_ldu:
                        ok, _ = self.validator.validate(h_ldu)
                        if ok:
                            ldus.append(h_ldu)
                if not seg_content.strip():
                    continue
                parent = section_stack[-1] if section_stack else None
                for para in re.split(r"\n\s*\n", seg_content):
                    para = para.strip()
                    if not para:
                        continue
                    chunk_type = "list" if _looks_like_list(para) else "paragraph"
                    token_count = count_tokens(para)
                    if token_count > self.max_tokens and chunk_type == "list":
                        for sub in _split_list_by_items(para, self.max_tokens):
                            ldu = self._make_text_ldu(
                                doc_id, sub, "list", [page_num], None, parent, idx_by_type
                            )
                            if ldu:
                                ok, _ = self.validator.validate(ldu)
                                if ok:
                                    ldus.append(ldu)
                    elif token_count > self.max_tokens:
                        for sub in self._split_long_text(para):
                            ldu = self._make_text_ldu(
                                doc_id, sub, "paragraph", [page_num], None, parent, idx_by_type
                            )
                            if ldu:
                                ok, _ = self.validator.validate(ldu)
                                if ok:
                                    ldus.append(ldu)
                    else:
                        ldu = self._make_text_ldu(
                            doc_id, para, chunk_type, [page_num], None, parent, idx_by_type
                        )
                        if ldu:
                            ok, _ = self.validator.validate(ldu)
                            if ok:
                                ldus.append(ldu)
        else:
            for page in doc.pages:
                text = page.text or ""
                if not text.strip():
                    continue
                for para in re.split(r"\n\s*\n", text):
                    para = para.strip()
                    if not para:
                        continue
                    # Detect heading as a standalone paragraph OR as the first line
                    # of a larger block (common in plain-text PDFs where there is
                    # no blank line between the heading and its body).
                    if _looks_like_plain_heading(para):
                        section_stack = [para.rstrip(":").strip()]
                        h_ldu = self._make_text_ldu(
                            doc_id, para, "heading", [page.page_number], None, None, idx_by_type
                        )
                        if h_ldu:
                            ok, _ = self.validator.validate(h_ldu)
                            if ok:
                                ldus.append(h_ldu)
                        continue
                    first_line, _, body = para.partition("\n")
                    if body.strip() and _looks_like_plain_heading(first_line.strip()):
                        section_stack = [first_line.strip().rstrip(":").strip()]
                        h_ldu = self._make_text_ldu(
                            doc_id, first_line.strip(), "heading", [page.page_number], None, None, idx_by_type
                        )
                        if h_ldu:
                            ok, _ = self.validator.validate(h_ldu)
                            if ok:
                                ldus.append(h_ldu)
                        para = body.strip()
                    parent = section_stack[-1] if section_stack else None
                    chunk_type = "list" if _looks_like_list(para) else "paragraph"
                    token_count = count_tokens(para)
                    if token_count > self.max_tokens and chunk_type == "list":
                        for sub in _split_list_by_items(para, self.max_tokens):
                            ldu = self._make_text_ldu(
                                doc_id, sub, "list", [page.page_number], None, parent, idx_by_type
                            )
                            if ldu:
                                ok, _ = self.validator.validate(ldu)
                                if ok:
                                    ldus.append(ldu)
                    elif token_count > self.max_tokens:
                        for sub in self._split_long_text(para):
                            ldu = self._make_text_ldu(
                                doc_id, sub, "paragraph", [page.page_number], None, parent, idx_by_type
                            )
                            if ldu:
                                ok, _ = self.validator.validate(ldu)
                                if ok:
                                    ldus.append(ldu)
                    else:
                        ldu = self._make_text_ldu(
                            doc_id, para, chunk_type, [page.page_number], None, parent, idx_by_type
                        )
                        if ldu:
                            ok, _ = self.validator.validate(ldu)
                            if ok:
                                ldus.append(ldu)

        return ldus

    def _split_long_text(self, text: str) -> list[str]:
        """Split text into chunks under max_tokens."""
        out: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for line in text.split("\n"):
            line_tokens = count_tokens(line)
            if current_tokens + line_tokens > self.max_tokens and current:
                out.append("\n".join(current))
                current = [line]
                current_tokens = line_tokens
            else:
                current.append(line)
                current_tokens += line_tokens
        if current:
            out.append("\n".join(current))
        return out if out else [text]

    def _make_text_ldu(
        self,
        doc_id: str,
        content: str,
        chunk_type: ChunkType,
        page_refs: list[int],
        bbox: BoundingBox | None,
        parent_section: str | None,
        idx_by_type: dict[ChunkType, int],
    ) -> LDU | None:
        if not content.strip():
            return None
        idx = idx_by_type.get(chunk_type, 0)
        idx_by_type[chunk_type] = idx + 1
        ldu_id = f"{doc_id}_{chunk_type}_{idx}"
        token_count = count_tokens(content)
        content_hash = _content_hash(content, page_refs, bbox)
        return LDU(
            ldu_id=ldu_id,
            content=content,
            chunk_type=chunk_type,
            page_refs=page_refs,
            bounding_box=bbox,
            parent_section=parent_section,
            token_count=token_count,
            content_hash=content_hash,
        )
