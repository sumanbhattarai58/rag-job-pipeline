"""Vector similarity retrieval using embeddings and Chroma."""
from app.embeddings.hf_embedder import HFEmbedder
from app.vectorstore.chroma_store import ChromaStore


class VectorRetriever:
    def __init__(self, embedder: HFEmbedder | None = None, store: ChromaStore | None = None):
        self.embedder = embedder or HFEmbedder()
        self.store = store or ChromaStore()

    def search(self, query: str, top_n: int, where: dict | None = None) -> list[dict]:
        query_embedding = self.embedder.embed([query])[0]
        result = self.store.query(embedding=query_embedding, top_n=top_n, where=where)

        ids = result["ids"][0]
        return [
            {
                "chunk_id": ids[i],
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "score": 1 - result["distances"][0][i],
            }
            for i in range(len(ids))
        ]