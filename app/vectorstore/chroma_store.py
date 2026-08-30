"""Chroma vector store wrapper."""
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings


class ChromaStore:
    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None):
        self.client = chromadb.PersistentClient(
            path=persist_dir or settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            collection_name or settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]):
        self.collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, embedding: list[float], top_n: int, where: dict | None = None) -> dict:
        return self.collection.query(query_embeddings=[embedding], n_results=top_n, where=_to_chroma_where(where))

    def count(self) -> int:
        return self.collection.count()


def _to_chroma_where(where: dict | None) -> dict | None:
    """Chroma requires multi-field filters wrapped in '$and'; a single field is passed as-is."""
    if not where:
        return None
    if len(where) == 1:
        return where
    return {"$and": [{key: value} for key, value in where.items()]}