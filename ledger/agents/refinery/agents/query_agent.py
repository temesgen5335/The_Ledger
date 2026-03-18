"""Query Interface Agent — LangGraph agent with three tools for document Q&A.

Tools:
  1. pageindex_navigate — section-level traversal via PageIndex
  2. semantic_search    — vector retrieval over LDU embeddings (ChromaDB)
  3. structured_query   — SQL over extracted fact tables (SQLite)

Every answer carries a ProvenanceChain with source citations.
Includes Audit Mode: claim verification that returns verified sources or 'unverifiable'.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph

from ledger.agents.refinery.agents.fact_table import FactTableExtractor
from ledger.agents.refinery.agents.indexer import load_page_index, query_page_index
from ledger.agents.refinery.agents.vector_store import LDUVectorStore
from ledger.agents.refinery.models.page_index import PageIndex
from ledger.agents.refinery.models.provenance import ProvenanceChain, SourceCitation
from ledger.agents.refinery.utils.llm import llm_generate

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _llm_call(prompt: str, api_key: str | None = None) -> str:
    """Generate a response using the first available LLM provider.

    The api_key parameter is accepted for backwards-compatibility but ignored —
    provider selection and keys are handled by src.utils.llm.llm_generate.
    """
    result = llm_generate(prompt)
    if result is None:
        return "(No LLM provider available — returning raw context)"
    return result


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def pageindex_navigate(
    topic: str,
    doc_id: str,
    pageindex_dir: Path | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Navigate the PageIndex tree for a document by topic. Returns top-k sections."""
    pi_dir = pageindex_dir or REPO_ROOT / ".refinery" / "pageindex"
    path = pi_dir / f"{doc_id}.json"
    if not path.exists():
        return []
    index = load_page_index(path)
    sections = query_page_index(index, topic, top_k=top_k)
    return [
        {
            "title": s.title,
            "page_start": s.page_start,
            "page_end": s.page_end,
            "summary": s.summary,
            "key_entities": s.key_entities,
            "data_types_present": s.data_types_present,
        }
        for s in sections
    ]


def semantic_search(
    query_text: str,
    doc_id: str | None = None,
    section_filter: str | None = None,
    store: LDUVectorStore | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Vector search over LDU embeddings. Returns chunks with metadata."""
    if store is None:
        chroma_dir = REPO_ROOT / ".refinery" / "chroma"
        store = LDUVectorStore(persist_directory=chroma_dir)
    return store.query(query_text, doc_id=doc_id, section_filter=section_filter, top_k=top_k)


def structured_query(
    keyword: str,
    doc_id: str | None = None,
    fact_table: FactTableExtractor | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """SQL query over the extracted fact table. Searches by keyword in fact_key."""
    if fact_table is None:
        db_path = REPO_ROOT / ".refinery" / "facts.db"
        fact_table = FactTableExtractor(db_path=db_path)
    return fact_table.query_facts(doc_id=doc_id, keyword=keyword, limit=limit)


# ---------------------------------------------------------------------------
# Provenance construction
# ---------------------------------------------------------------------------

def _build_provenance_from_results(
    answer: str,
    search_results: list[dict[str, Any]],
    fact_results: list[dict[str, Any]] | None = None,
) -> ProvenanceChain:
    """Construct ProvenanceChain from search and fact results."""
    sources: list[SourceCitation] = []

    for r in search_results:
        meta = r.get("metadata", {})
        page_refs_str = meta.get("page_refs", "")
        pages = [int(p) for p in page_refs_str.split(",") if p.strip().isdigit()]
        page_num = pages[0] if pages else 1
        sources.append(SourceCitation(
            document_name=meta.get("doc_id", "unknown"),
            page_number=page_num,
            bounding_box=None,
            content_hash=meta.get("content_hash", ""),
            excerpt=r.get("content", "")[:200],
        ))

    if fact_results:
        for f in fact_results:
            sources.append(SourceCitation(
                document_name=f.get("doc_id", "unknown"),
                page_number=f.get("page_ref", 1) or 1,
                bounding_box=None,
                content_hash=f.get("content_hash", ""),
                excerpt=f"{f.get('fact_key', '')}: {f.get('fact_value', '')} {f.get('unit', '')}".strip(),
            ))

    return ProvenanceChain(answer=answer, sources=sources)


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

# ChromaDB returns L2 distance by default (0 = identical vectors).
# Results above this threshold are too dissimilar to count as supporting evidence.
_AUDIT_DISTANCE_THRESHOLD = 1.0

_STOP_WORDS = frozenset({
    "the", "was", "were", "has", "have", "had", "been", "are", "is", "and",
    "that", "this", "for", "from", "with", "not", "but", "its", "our", "their",
    "per", "all", "any", "more", "also", "than", "into", "over", "both",
})

_NUMBER_RE = re.compile(
    r"""
    (?<!\w)                          # not preceded by a word char
    (\d{1,3}(?:[,\s]\d{3})*         # integer with optional thousands separators
    (?:\.\d+)?                       # optional decimal
    |\d+\.\d+                        # plain decimal
    |\d+)                            # plain integer
    (?!\w)                           # not followed by a word char
    """,
    re.VERBOSE,
)


def _extract_numbers(text: str) -> list[float]:
    """Return all numeric values found in *text* (strips thousands separators)."""
    out: list[float] = []
    for m in _NUMBER_RE.finditer(text):
        try:
            out.append(float(m.group(0).replace(",", "").replace(" ", "")))
        except ValueError:
            pass
    return out


def _numerical_fact_check(
    claim_numbers: list[float],
    fact_results: list[dict[str, Any]],
    tolerance: float = 0.01,
) -> str | None:
    """Compare numbers extracted from the claim against fact-table values.

    Returns:
        ``"CONTRADICTED"`` — at least one fact-table value conflicts with a
        claim number (difference > tolerance fraction of the fact value).
        ``"SUPPORTED"``    — at least one fact-table value exactly matches a
        claim number.
        ``None``           — inconclusive (no overlap found either way).
    """
    if not claim_numbers or not fact_results:
        return None

    matched = False
    for fact in fact_results:
        raw = fact.get("fact_value")
        if raw is None:
            continue
        try:
            fact_val = float(str(raw).replace(",", ""))
        except (ValueError, TypeError):
            continue

        for cn in claim_numbers:
            # Avoid division by zero for near-zero facts
            denom = abs(fact_val) if abs(fact_val) > 1e-9 else 1.0
            if abs(cn - fact_val) / denom <= tolerance:
                matched = True
            elif abs(cn - fact_val) / denom > tolerance and abs(cn - fact_val) > 0.5:
                # Only flag as contradicted if the discrepancy is meaningful
                return "CONTRADICTED"

    return "SUPPORTED" if matched else None


# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    question: str
    doc_id: str
    tool_results: dict[str, Any]
    answer: str
    provenance: dict[str, Any]


# ---------------------------------------------------------------------------
# Query Agent class
# ---------------------------------------------------------------------------

class QueryAgent:
    """LangGraph-based query agent with three tools for document Q&A."""

    def __init__(
        self,
        store: LDUVectorStore | None = None,
        fact_table: FactTableExtractor | None = None,
        pageindex_dir: Path | str | None = None,
        api_key: str | None = None,
    ):
        chroma_dir = REPO_ROOT / ".refinery" / "chroma"
        self.store = store or LDUVectorStore(persist_directory=chroma_dir)
        self.fact_table = fact_table or FactTableExtractor()
        self.pageindex_dir = Path(pageindex_dir) if pageindex_dir else REPO_ROOT / ".refinery" / "pageindex"
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or ""
        self._graph = self._build_graph()

    def _classify_query(self, question: str) -> Literal["navigate", "numerical", "factual"]:
        """Route the query to the appropriate tool based on question type."""
        q = question.lower()
        numerical_signals = [
            "how much", "how many", "what is the", "total", "revenue", "cost",
            "amount", "percentage", "rate", "number of", "value of", "price",
            "budget", "expenditure", "income", "profit", "loss",
        ]
        navigational_signals = [
            "where", "which section", "find the section", "locate",
            "table of contents", "what sections", "overview of",
        ]

        if any(s in q for s in navigational_signals):
            return "navigate"
        if any(s in q for s in numerical_signals):
            return "numerical"
        return "factual"

    def _run_tools(self, state: AgentState) -> AgentState:
        """Execute tools based on query classification."""
        question = state["question"]
        doc_id = state["doc_id"]
        query_type = self._classify_query(question)
        tool_results: dict[str, Any] = {"query_type": query_type}

        if query_type == "navigate":
            sections = pageindex_navigate(question, doc_id, self.pageindex_dir)
            tool_results["pageindex"] = sections
            if sections:
                section_title = sections[0].get("title")
                tool_results["semantic"] = semantic_search(
                    question, doc_id=doc_id, section_filter=section_title, store=self.store, top_k=3,
                )
            else:
                tool_results["semantic"] = semantic_search(
                    question, doc_id=doc_id, store=self.store, top_k=5,
                )
        elif query_type == "numerical":
            tool_results["facts"] = structured_query(
                keyword=question, doc_id=doc_id, fact_table=self.fact_table,
            )
            tool_results["semantic"] = semantic_search(
                question, doc_id=doc_id, store=self.store, top_k=3,
            )
        else:
            tool_results["semantic"] = semantic_search(
                question, doc_id=doc_id, store=self.store, top_k=5,
            )
            sections = pageindex_navigate(question, doc_id, self.pageindex_dir, top_k=2)
            tool_results["pageindex"] = sections

        return {**state, "tool_results": tool_results}

    def _generate_answer(self, state: AgentState) -> AgentState:
        """Synthesize an answer from tool results and build provenance chain."""
        question = state["question"]
        tool_results = state["tool_results"]

        context_parts: list[str] = []

        if "pageindex" in tool_results and tool_results["pageindex"]:
            for s in tool_results["pageindex"]:
                context_parts.append(
                    f"[Section: {s['title']}, pp. {s['page_start']}-{s['page_end']}] "
                    f"{s.get('summary', '') or ''}"
                )

        search_results = tool_results.get("semantic", [])
        for r in search_results:
            meta = r.get("metadata", {})
            context_parts.append(
                f"[Source: {meta.get('doc_id', '?')}, p. {meta.get('page_refs', '?')}, "
                f"type={meta.get('chunk_type', '?')}]\n{r.get('content', '')}"
            )

        fact_results = tool_results.get("facts", [])
        for f in fact_results:
            context_parts.append(
                f"[Fact: {f.get('fact_key', '')}: {f.get('fact_value', '')} {f.get('unit', '')} "
                f"(p. {f.get('page_ref', '?')}, doc: {f.get('doc_id', '?')})]"
            )

        context = "\n\n".join(context_parts) if context_parts else "(No relevant context found)"

        prompt = (
            f"You are a document intelligence assistant. Answer the question based ONLY on the "
            f"provided context. Cite page numbers when possible. If the answer is not in the "
            f"context, say 'The document does not contain this information.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )

        answer = _llm_call(prompt, self.api_key)

        provenance = _build_provenance_from_results(
            answer=answer,
            search_results=search_results,
            fact_results=fact_results,
        )

        return {
            **state,
            "answer": answer,
            "provenance": provenance.model_dump(),
        }

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph for the query agent."""
        graph = StateGraph(AgentState)
        graph.add_node("run_tools", self._run_tools)
        graph.add_node("generate_answer", self._generate_answer)
        graph.set_entry_point("run_tools")
        graph.add_edge("run_tools", "generate_answer")
        graph.add_edge("generate_answer", END)
        return graph.compile()

    def query(self, question: str, doc_id: str) -> ProvenanceChain:
        """Ask a question about a specific document. Returns answer with ProvenanceChain."""
        initial_state: AgentState = {
            "question": question,
            "doc_id": doc_id,
            "tool_results": {},
            "answer": "",
            "provenance": {},
        }
        result = self._graph.invoke(initial_state)
        return ProvenanceChain.model_validate(result["provenance"])

    def audit_claim(self, claim: str, doc_id: str) -> ProvenanceChain:
        """Audit Mode: verify a claim against the document.

        Three-stage pipeline:
        1. Retrieve — semantic search + fact-table lookup.
        2. Filter   — drop semantic results whose distance exceeds the relevance
                      threshold (too dissimilar to count as evidence).
        3. Decide   — deterministic numerical check first; LLM judgment with a
                      three-way verdict (VERIFIED / CONTRADICTED / UNVERIFIABLE)
                      when numbers alone are inconclusive.
        """
        # --- Stage 1: retrieval ---
        raw_search = semantic_search(claim, doc_id=doc_id, store=self.store, top_k=5)

        # Query fact table with individual meaningful tokens rather than the full
        # claim string, so "Revenue was 999 M" finds the fact keyed as "Revenue".
        _claim_keywords = [
            w for w in re.findall(r"\b[A-Za-z]{3,}\b", claim)
            if w.lower() not in _STOP_WORDS
        ]
        _seen_fact_ids: set[Any] = set()
        fact_results: list[dict[str, Any]] = []
        for kw in _claim_keywords[:6]:
            for row in structured_query(keyword=kw, doc_id=doc_id, fact_table=self.fact_table):
                row_id = row.get("id", id(row))
                if row_id not in _seen_fact_ids:
                    _seen_fact_ids.add(row_id)
                    fact_results.append(row)

        # --- Stage 2: filter by relevance distance ---
        relevant_search = [
            r for r in raw_search
            if (r.get("distance") is None or r["distance"] < _AUDIT_DISTANCE_THRESHOLD)
        ]

        has_evidence = bool(relevant_search) or bool(fact_results)
        if not has_evidence:
            verdict = (
                f"UNVERIFIABLE: no relevant evidence found in '{doc_id}' for this claim."
            )
            return _build_provenance_from_results(
                answer=verdict,
                search_results=[],
                fact_results=[],
            )

        # --- Stage 3a: deterministic numerical check ---
        claim_numbers = _extract_numbers(claim)
        numerical_result = _numerical_fact_check(claim_numbers, fact_results)

        if numerical_result == "CONTRADICTED":
            matched_facts = [
                f"{f.get('fact_key', '')}: {f.get('fact_value', '')} {f.get('unit', '')}".strip()
                for f in fact_results
            ]
            verdict = (
                f"CONTRADICTED: numbers in the claim do not match the document's fact table. "
                f"Document records: {'; '.join(matched_facts)}."
            )
            return _build_provenance_from_results(
                answer=verdict,
                search_results=relevant_search,
                fact_results=fact_results,
            )

        # --- Stage 3b: LLM judgment with three-way verdict ---
        context_parts: list[str] = []
        for r in relevant_search:
            meta = r.get("metadata", {})
            context_parts.append(
                f"[p.{meta.get('page_refs', '?')}] {r.get('content', '')}"
            )
        for f in fact_results:
            context_parts.append(
                f"[Fact] {f.get('fact_key', '')}: {f.get('fact_value', '')} "
                f"{f.get('unit', '')} (p.{f.get('page_ref', '?')})"
            )
        context = "\n".join(context_parts)

        prompt = (
            "You are a document auditor. Verify the following claim strictly against "
            "the provided evidence from the document. Choose exactly one verdict:\n"
            "  VERIFIED     — the evidence directly and explicitly supports the claim.\n"
            "  CONTRADICTED — the evidence directly contradicts the claim (wrong value, "
            "wrong date, wrong entity).\n"
            "  UNVERIFIABLE — the evidence is related but does not confirm or deny the claim.\n\n"
            f"Claim: {claim}\n\n"
            f"Evidence:\n{context}\n\n"
            "Respond with exactly: VERIFIED: <explanation>  or  CONTRADICTED: <explanation>  "
            "or  UNVERIFIABLE: <explanation>"
        )
        verdict = _llm_call(prompt, self.api_key)

        return _build_provenance_from_results(
            answer=verdict,
            search_results=relevant_search,
            fact_results=fact_results,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.agents.query_agent <doc_id> <question>", file=sys.stderr)
        print("       python -m src.agents.query_agent --audit <doc_id> <claim>", file=sys.stderr)
        sys.exit(1)

    if sys.argv[1] == "--audit":
        if len(sys.argv) < 4:
            print("Usage: python -m src.agents.query_agent --audit <doc_id> <claim>", file=sys.stderr)
            sys.exit(1)
        doc = sys.argv[2]
        claim_text = " ".join(sys.argv[3:])
        agent = QueryAgent()
        result = agent.audit_claim(claim_text, doc)
    else:
        doc = sys.argv[1]
        q = " ".join(sys.argv[2:])
        agent = QueryAgent()
        result = agent.query(q, doc)

    print(json.dumps(result.model_dump(), indent=2))
