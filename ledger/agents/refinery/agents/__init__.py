"""Pipeline agents — triage, extraction router, and downstream stages."""

from ledger.agents.refinery.agents.triage import TriageAgent

# Lazy import so `python -m src.agents.extractor` loads the module as __main__ first
def __getattr__(name: str):
    if name == "ExtractionRouter":
        from src.agents.extractor import ExtractionRouter
        return ExtractionRouter
    if name == "ChunkingEngine" or name == "ChunkValidator":
        from src.agents.chunker import ChunkingEngine, ChunkValidator
        return ChunkingEngine if name == "ChunkingEngine" else ChunkValidator
    if name == "PageIndexBuilder":
        from src.agents.indexer import PageIndexBuilder
        return PageIndexBuilder
    if name == "LDUVectorStore":
        from src.agents.vector_store import LDUVectorStore
        return LDUVectorStore
    if name == "FactTableExtractor":
        from src.agents.fact_table import FactTableExtractor
        return FactTableExtractor
    if name == "QueryAgent":
        from src.agents.query_agent import QueryAgent
        return QueryAgent
    if name == "PageAnalyzer":
        from src.agents.page_analyzer import PageAnalyzer
        return PageAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ExtractionRouter",
    "TriageAgent",
    "PageAnalyzer",
    "ChunkingEngine",
    "ChunkValidator",
    "PageIndexBuilder",
    "LDUVectorStore",
    "FactTableExtractor",
    "QueryAgent",
]
