"""PageIndex tree builder — section hierarchy and LLM summaries for navigation."""

import json
import os
import re
from pathlib import Path
from typing import Callable

from ledger.agents.refinery.models.ldu import LDU
from ledger.agents.refinery.models.page_index import DataType, PageIndex, Section
from ledger.agents.refinery.utils.llm import llm_generate


SECTION_SUMMARY_PROMPT = (
    "Summarize the following document section in 2-3 concise sentences. "
    "Focus on key facts, figures, and conclusions. Output only the summary, no preamble."
)

# Named entity extraction patterns (lightweight, no spaCy dependency)
_ORG_RE = re.compile(
    r"\b(?:Ministry of|Bank of|Government of|Department of|Authority|Commission|"
    r"Corporation|Institute|University|Agency|Office of|Bureau of|"
    r"National|Federal|Commercial Bank|Development Bank|Central Bank|"
    r"United Nations|World Bank|IMF|WTO|AU|EU)"
    r"[\w\s,&]{0,60}?\b",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(
    r"(?:ETB|USD|EUR|GBP|Birr|\$|€|£)\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|thousand|mn|bn|m|k)?",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2})"
    r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}"
    r"|FY\s*\d{4}(?:[/-]\d{2,4})?"
    r"|\b(?:Q[1-4])\s+\d{4}",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}(?:[/\-–](?:19|20)?\d{2})?\b")

# Capitalized multi-word phrases (proper nouns / project names)
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:[ \-][A-Z][a-z]+)+\b")
# "Project <Name>" pattern
_PROJECT_RE = re.compile(r"\bProject[ ]+[\w\-]+\b", re.IGNORECASE)
# Hyphenated compound terms (e.g. Fortune-Connect, Better-Auth)
_HYPHENATED_RE = re.compile(r"\b[A-Z][\w]*(?:-[A-Z][\w]*)+\b")
# PascalCase / camelCase technical terms (e.g. SpecKit, DrizzleKit)
_TECH_TERM_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "and", "but", "or",
    "nor", "not", "so", "yet", "for", "with", "from", "into", "that",
    "this", "these", "those", "what", "which", "who", "whom", "how",
    "when", "where", "why", "all", "each", "both", "few", "more", "most",
    "other", "some", "such", "than", "too", "very", "also", "just", "about",
    "above", "after", "before", "between", "here", "there", "then", "once",
    "been", "its", "our", "their", "your", "his", "her", "only", "same",
    "still", "while", "during", "none",
})


def _extract_key_entities(text: str, max_entities: int = 15) -> list[str]:
    """Extract named entities and key terms from text.

    Uses structured regex patterns first (orgs, money, dates), then
    falls back to proper nouns, project names, hyphenated compounds,
    and PascalCase terms to ensure sections have enough keywords for
    topic-based retrieval.
    """
    if not text or len(text.strip()) < 10:
        return []
    entities: list[str] = []
    seen_lower: set[str] = set()

    def _add(entity: str) -> bool:
        entity = entity.strip().rstrip(",.")
        if len(entity) < 3 or entity.lower() in seen_lower:
            return False
        if entity.lower() in _STOPWORDS:
            return False
        seen_lower.add(entity.lower())
        entities.append(entity)
        return len(entities) >= max_entities

    for pattern in (_ORG_RE, _MONEY_RE, _DATE_RE, _YEAR_RE):
        for m in pattern.finditer(text):
            if _add(m.group(0)):
                return entities

    for pattern in (_PROJECT_RE, _HYPHENATED_RE, _PROPER_NOUN_RE, _TECH_TERM_RE):
        for m in pattern.finditer(text):
            if _add(m.group(0)):
                return entities

    return entities


def _summarize_with_llm(text: str, api_key: str | None = None) -> str | None:
    """Generate a 2-3 sentence section summary via the first available LLM provider.

    Provider selection, retry logic, and circuit-breaking are handled by
    src.utils.llm.llm_generate.  The api_key parameter is accepted for
    backwards-compatibility but ignored — keys are read from env vars.
    """
    if not text or not text.strip() or len(text.strip()) < 20:
        return None
    prompt = SECTION_SUMMARY_PROMPT + "\n\n---\n\n" + text[:12000]
    return llm_generate(prompt)


def _data_types_from_ldus(ldus: list[LDU]) -> list[DataType]:
    """Infer data_types_present from LDU chunk_types."""
    present = set()
    for ldu in ldus:
        if ldu.chunk_type == "table":
            present.add("tables")
        elif ldu.chunk_type == "figure":
            present.add("figures")
    return list(present)


def _build_section_tree_from_ldus(
    doc_id: str,
    ldus: list[LDU],
    summarize_fn: Callable[[str, str | None], str | None] | None = None,
    api_key: str | None = None,
) -> PageIndex:
    """Build PageIndex from LDUs. Groups by parent_section; each section gets page range and LLM summary."""
    by_section: dict[str | None, list[LDU]] = {}
    for ldu in ldus:
        key = ldu.parent_section if ldu.parent_section else None
        by_section.setdefault(key, []).append(ldu)

    def make_section(section_title: str | None, section_ldus: list[LDU]) -> Section:
        if not section_ldus:
            pages = [1]
        else:
            pages = []
            for l in section_ldus:
                pages.extend(l.page_refs)
            pages = sorted(set(pages)) if pages else [1]
        page_start = min(pages)
        page_end = max(pages)
        content = "\n\n".join(l.content for l in section_ldus if l.content.strip())
        summary = None
        fn = summarize_fn or _summarize_with_llm
        if content:
            summary = fn(content, api_key)
        data_types = _data_types_from_ldus(section_ldus)
        key_entities = _extract_key_entities(content)
        return Section(
            title=section_title or "Document",
            page_start=page_start,
            page_end=page_end,
            child_sections=[],
            key_entities=key_entities,
            summary=summary,
            data_types_present=data_types,
        )

    child_sections: list[Section] = []
    for key, section_ldus in by_section.items():
        if key is None:
            continue
        child_sections.append(make_section(key, section_ldus))

    root_ldus = by_section.get(None, [])
    all_pages = []
    for l in ldus:
        all_pages.extend(l.page_refs)
    root_page_start = min(all_pages) if all_pages else 1
    root_page_end = max(all_pages) if all_pages else 1
    all_content = "\n\n".join(l.content for l in ldus if l.content.strip())
    root_summary = _summarize_with_llm(all_content[:8000], api_key) if all_content else None
    root_entities = _extract_key_entities(all_content)
    root_section = Section(
        title=doc_id,
        page_start=root_page_start,
        page_end=root_page_end,
        child_sections=child_sections,
        key_entities=root_entities,
        summary=root_summary,
        data_types_present=_data_types_from_ldus(ldus),
    )

    return PageIndex(doc_id=doc_id, root_section=root_section)


class PageIndexBuilder:
    """Builds PageIndex from LDUs and persists to .refinery/pageindex/."""

    def __init__(
        self,
        output_dir: Path | str | None = None,
        api_key: str | None = None,
        summarize_fn: Callable[[str, str | None], str | None] | None = None,
    ):
        base = Path(__file__).resolve().parent.parent.parent
        self.output_dir = Path(output_dir) if output_dir else base / ".refinery" / "pageindex"
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or ""
        self.summarize_fn = summarize_fn or _summarize_with_llm

    def build(self, doc_id: str, ldus: list[LDU]) -> PageIndex:
        """Build PageIndex from LDUs and return it."""
        return _build_section_tree_from_ldus(
            doc_id,
            ldus,
            summarize_fn=self.summarize_fn,
            api_key=self.api_key or None,
        )

    def build_and_save(self, doc_id: str, ldus: list[LDU]) -> Path:
        """Build PageIndex, save to output_dir/{doc_id}.json, return path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        index = self.build(doc_id, ldus)
        path = self.output_dir / f"{doc_id}.json"
        with open(path, "w") as f:
            json.dump(index.model_dump(), f, indent=2)
        return path


def load_page_index(path: Path | str) -> PageIndex:
    """Load PageIndex from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return PageIndex.model_validate(data)


_PUNCT_RE = re.compile(r"[^\w\s\-]")


def _normalise(text: str) -> str:
    """Lowercase and strip punctuation (except hyphens) for scoring."""
    return _PUNCT_RE.sub("", text.lower())


def query_page_index(
    index: PageIndex,
    topic: str,
    top_k: int = 3,
) -> list[Section]:
    """Traverse PageIndex to return top-k sections most relevant to topic."""
    topic_norm = _normalise(topic)
    topic_words = set(topic_norm.split()) - _STOPWORDS

    def score_section(section: Section) -> float:
        text_norm = _normalise(f"{section.title} {section.summary or ''}")
        text_words = set(text_norm.split()) - _STOPWORDS
        word_hits = len(topic_words & text_words)

        entity_bonus = 0.0
        if section.key_entities:
            for entity in section.key_entities:
                ent_norm = _normalise(entity)
                if ent_norm in topic_norm:
                    entity_bonus += 2.0
                elif any(w in topic_words for w in ent_norm.split() if w not in _STOPWORDS):
                    entity_bonus += 0.5

        substring_bonus = 0.5 if topic_norm in _normalise(
            f"{section.title} {section.summary or ''} "
            + " ".join(section.key_entities or [])
        ) else 0.0

        return word_hits + entity_bonus + substring_bonus

    def flatten_sections(section: Section) -> list[Section]:
        out = [section]
        for child in section.child_sections:
            out.extend(flatten_sections(child))
        return out

    all_sections = flatten_sections(index.root_section)
    scored = [(score_section(s), s) for s in all_sections]
    scored.sort(key=lambda x: -x[0])
    return [s for score, s in scored[:top_k] if score > 0]


def precision_with_page_index(
    index: PageIndex,
    topic: str,
    gold_section_titles: set[str] | None = None,
    gold_pages: set[int] | None = None,
    top_k: int = 3,
) -> tuple[bool, list[Section]]:
    """Return (hit, top_sections): whether any gold section or page is in the top-k sections."""
    top_sections = query_page_index(index, topic, top_k=top_k)
    hit = False
    if gold_section_titles:
        for s in top_sections:
            if s.title in gold_section_titles:
                hit = True
                break
    if gold_pages and not hit:
        for s in top_sections:
            if any(p in gold_pages for p in range(s.page_start, s.page_end + 1)):
                hit = True
                break
    if not gold_section_titles and not gold_pages:
        hit = len(top_sections) > 0
    return hit, top_sections
