"""Vector store ingestion for LDUs — ChromaDB with doc_id and section filtering."""

from pathlib import Path
from typing import Any

from ledger.agents.refinery.models.ldu import LDU


def _get_client(persist_directory: Path | str | None = None):
    """Return ChromaClient; persist to directory if given."""
    import chromadb

    if persist_directory is None:
        return chromadb.EphemeralClient()
    return chromadb.PersistentClient(path=str(persist_directory))


def _ldu_to_metadata(ldu: LDU, doc_id: str) -> dict[str, Any]:
    """Chroma metadata: strings/ints only; page_refs as comma-separated string."""
    return {
        "doc_id": doc_id,
        "ldu_id": ldu.ldu_id,
        "chunk_type": ldu.chunk_type,
        "page_refs": ",".join(str(p) for p in ldu.page_refs) if ldu.page_refs else "",
        "parent_section": ldu.parent_section or "",
        "content_hash": ldu.content_hash,
    }


class LDUVectorStore:
    """ChromaDB-backed store for LDU embeddings; supports ingest and query with optional section filter."""

    def __init__(
        self,
        collection_name: str = "refinery_ldus",
        persist_directory: Path | str | None = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory) if persist_directory else None
        self._client = _get_client(self.persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Refinery LDUs for RAG"},
        )

    def ingest_ldus(self, doc_id: str, ldus: list[LDU]) -> None:
        """Insert LDUs for a document. Uses LDU content for embedding; metadata for filtering."""
        if not ldus:
            return
        ids = [ldu.ldu_id for ldu in ldus]
        documents = [ldu.content for ldu in ldus]
        metadatas = [_ldu_to_metadata(ldu, doc_id) for ldu in ldus]
        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def query(
        self,
        query_text: str,
        doc_id: str | None = None,
        section_filter: str | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Semantic search over LDUs. Optionally restrict to doc_id and/or parent_section."""
        where: dict[str, Any] | None = None
        if doc_id and section_filter:
            where = {"$and": [{"doc_id": doc_id}, {"parent_section": section_filter}]}
        elif doc_id:
            where = {"doc_id": doc_id}
        elif section_filter:
            where = {"parent_section": section_filter}
        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where
        result = self._collection.query(**kwargs)
        out: list[dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances") or []
        dist_list = (dists[0] if dists else []) or []
        for i, id_ in enumerate(ids):
            out.append({
                "ldu_id": id_,
                "content": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dist_list[i] if i < len(dist_list) else None,
            })
        return out
