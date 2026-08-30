"""Hybrid retrieval: combines vector similarity and BM25 keyword scores."""
from app.config import settings
from app.retrieval.keyword_retriever import KeywordRetriever
from app.retrieval.vector_retriever import VectorRetriever


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {chunk_id: 1.0 for chunk_id in scores}
    return {chunk_id: (v - lo) / (hi - lo) for chunk_id, v in scores.items()}


class HybridRetriever:
    def __init__(
        self,
        vector_retriever: VectorRetriever | None = None,
        keyword_retriever: KeywordRetriever | None = None,
    ):
        self.vector_retriever = vector_retriever or VectorRetriever()
        self.keyword_retriever = keyword_retriever or KeywordRetriever()

    def search(self, query: str, top_n: int, where: dict | None = None) -> list[dict]:
        candidate_pool = max(top_n * 4, 20)
        vector_hits = self.vector_retriever.search(query, candidate_pool, where=where)
        keyword_hits = self.keyword_retriever.search(query, candidate_pool, where=where)

        vector_scores = _normalize({h["chunk_id"]: h["score"] for h in vector_hits})
        keyword_scores = _normalize({h["chunk_id"]: h["score"] for h in keyword_hits})

        by_id = {h["chunk_id"]: h for h in vector_hits}
        for hit in keyword_hits:
            by_id.setdefault(hit["chunk_id"], hit)

        combined = []
        for chunk_id, hit in by_id.items():
            score = (
                settings.vector_weight * vector_scores.get(chunk_id, 0.0)
                + settings.bm25_weight * keyword_scores.get(chunk_id, 0.0)
            )
            combined.append({**hit, "score": score})

        combined.sort(key=lambda h: h["score"], reverse=True)
        return combined[:top_n]