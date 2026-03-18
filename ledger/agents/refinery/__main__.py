"""PaperMind CLI — end-to-end document intelligence pipeline.

Usage:
  python -m src ingest  [--force] <pdf> [<pdf> ...]
  python -m src query   <doc_id> <topic>
  python -m src ask     <doc_id> <question>
  python -m src audit   <doc_id> <claim>

Or, if installed via pip/uv:
  papermind ingest  [--force] <pdf> [<pdf> ...]
  papermind query   <doc_id> <topic>
  papermind ask     <doc_id> <question>
  papermind audit   <doc_id> <claim>
"""

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Load .env before any langchain/langgraph imports so tracing env vars are set
# before LangSmith initialises its callback handlers.
try:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
except ImportError:
    pass

USAGE = """\
papermind — Document Intelligence Pipeline

Subcommands:
  ingest  [--force] <pdf> [<pdf> ...]   Triage → extract → chunk → index → store
  query   <doc_id> <topic>              PageIndex topic lookup
  ask     <doc_id> <question>           Natural-language Q&A
  audit   <doc_id> <claim>              Claim verification with provenance
"""


def _resolve(path_arg: str) -> Path:
    p = Path(path_arg)
    return p if p.is_absolute() else REPO / p


# ------------------------------------------------------------------
# Subcommands
# ------------------------------------------------------------------

def cmd_ingest(argv: list[str]) -> None:
    """Triage → extract → chunk → PageIndex → vector store → fact table."""
    from src.agents.chunker import ChunkingEngine
    from src.agents.extractor import ExtractionRouter
    from src.agents.fact_table import FactTableExtractor
    from src.agents.indexer import PageIndexBuilder
    from src.agents.triage import TriageAgent
    from src.agents.vector_store import LDUVectorStore
    from src.models.document_profile import DocumentProfile
    from src.utils.doc_id import stem_to_doc_id

    force = "--force" in argv
    pdf_args = [a for a in argv if a != "--force"]

    if not pdf_args:
        print("Usage: papermind ingest [--force] <pdf> [<pdf> ...]", file=sys.stderr)
        sys.exit(1)

    profiles_dir = REPO / ".refinery" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    pageindex_dir = REPO / ".refinery" / "pageindex"
    chroma_dir = REPO / ".refinery" / "chroma"
    stats_dir = REPO / ".refinery" / "output" / "output_pdfplumber"

    triage = TriageAgent()
    router = ExtractionRouter(
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        stats_dir=stats_dir,
    )
    chunker = ChunkingEngine()
    indexer = PageIndexBuilder(output_dir=pageindex_dir, api_key=os.environ.get("GEMINI_API_KEY"))
    store = LDUVectorStore(collection_name="refinery_ldus", persist_directory=chroma_dir)
    fact_extractor = FactTableExtractor()

    for pdf_arg in pdf_args:
        path = _resolve(pdf_arg)
        if not path.exists():
            print(f"SKIP (not found): {path}")
            continue

        doc_id = stem_to_doc_id(path.stem)
        profile_file = profiles_dir / f"{doc_id}.json"
        if profile_file.exists():
            profile = DocumentProfile.model_validate_json(profile_file.read_text())
        else:
            profile = triage.profile_and_save(path, output_dir=profiles_dir)

        print(f"[{doc_id}] Extracting (force={force})...")
        try:
            extracted = router.extract(path, profile, force_reextract=force)
        except Exception as e:
            print(f"[{doc_id}] Extraction failed: {e}")
            continue
        print(f"[{doc_id}] strategy={extracted.strategy_used} conf={extracted.confidence_score:.2f}")

        print(f"[{doc_id}] Chunking...")
        ldus = chunker.chunk(extracted)
        print(f"[{doc_id}] LDUs: {len(ldus)}")
        if not ldus:
            print(f"[{doc_id}] No LDUs; skipping downstream stages.")
            continue

        print(f"[{doc_id}] Building PageIndex...")
        index_path = indexer.build_and_save(doc_id, ldus)
        print(f"[{doc_id}] PageIndex -> {index_path}")

        print(f"[{doc_id}] Ingesting into vector store...")
        store.ingest_ldus(doc_id, ldus)

        print(f"[{doc_id}] Extracting facts...")
        n_facts = fact_extractor.extract_and_store(doc_id, ldus)
        print(f"[{doc_id}] Facts: {n_facts}")
        print(f"[{doc_id}] Done.")


def cmd_query(argv: list[str]) -> None:
    """Navigate the PageIndex by topic."""
    from src.agents.indexer import load_page_index, query_page_index

    if len(argv) < 2:
        print("Usage: papermind query <doc_id> <topic>", file=sys.stderr)
        sys.exit(1)

    doc_id, topic = argv[0], " ".join(argv[1:])
    path = REPO / ".refinery" / "pageindex" / f"{doc_id}.json"
    if not path.exists():
        print(f"PageIndex not found: {path}", file=sys.stderr)
        sys.exit(1)

    index = load_page_index(path)
    top = query_page_index(index, topic, top_k=3)
    print(f"Top-3 sections for topic {topic!r}:")
    if top:
        for s in top:
            print(f"  - {s.title} (pp. {s.page_start}–{s.page_end})")
    else:
        print("  (no matching sections — listing all available sections)")

        def _all_sections(sec):  # type: ignore[no-untyped-def]
            yield sec
            for c in sec.child_sections:
                yield from _all_sections(c)

        for s in _all_sections(index.root_section):
            print(f"  - {s.title} (pp. {s.page_start}–{s.page_end})")


def cmd_ask(argv: list[str]) -> None:
    """Natural-language Q&A via the Query Agent."""
    from src.agents.query_agent import QueryAgent

    if len(argv) < 2:
        print("Usage: papermind ask <doc_id> <question>", file=sys.stderr)
        sys.exit(1)

    doc_id = argv[0]
    question = " ".join(argv[1:])
    agent = QueryAgent()
    result = agent.query(question, doc_id)
    print(json.dumps(result.model_dump(), indent=2))


def cmd_audit(argv: list[str]) -> None:
    """Claim verification with provenance chain."""
    from src.agents.query_agent import QueryAgent

    if len(argv) < 2:
        print("Usage: papermind audit <doc_id> <claim>", file=sys.stderr)
        sys.exit(1)

    doc_id = argv[0]
    claim = " ".join(argv[1:])
    agent = QueryAgent()
    result = agent.audit_claim(claim, doc_id)
    print(json.dumps(result.model_dump(), indent=2))


# ------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------

COMMANDS = {
    "ingest": cmd_ingest,
    "query": cmd_query,
    "ask": cmd_ask,
    "audit": cmd_audit,
}


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0)

    cmd_name = args[0]
    handler = COMMANDS.get(cmd_name)
    if handler is None:
        print(f"Unknown command: {cmd_name!r}\n", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        sys.exit(1)

    handler(args[1:])


if __name__ == "__main__":
    main()
