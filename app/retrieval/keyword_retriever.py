"""BM25 keyword retrieval over the full chunk corpus."""
from rank_bm25 import BM25Okapi

from app.config import settings
from app.ingestion.loader import load_jobs
from app.ingestion.pipeline import JobChunkRecord, build_records, record_to_metadata


class KeywordRetriever:
    def __init__(self, records: list[JobChunkRecord] | None = None):
        self.records = records or self._load_records()
        self.bm25 = BM25Okapi([r.text.lower().split() for r in self.records])

    def _load_records(self) -> list[JobChunkRecord]:
        df = load_jobs(settings.data_path)
        return build_records(df, settings.chunk_size_tokens, settings.chunk_overlap_tokens)

    def search(self, query: str, top_n: int, where: dict | None = None) -> list[dict]:
        scores = self.bm25.get_scores(query.lower().split())

        indices = range(len(scores))
        if where:
            indices = [i for i in indices if self._matches(self.records[i], where)]

        ranked = sorted(indices, key=lambda i: scores[i], reverse=True)[:top_n]
        return [
            {
                "chunk_id": self.records[i].chunk_id,
                "text": self.records[i].text,
                "metadata": record_to_metadata(self.records[i]),
                "score": float(scores[i]),
            }
            for i in ranked
        ]

    def _matches(self, record: JobChunkRecord, where: dict) -> bool:
        metadata = record_to_metadata(record)
        return all(metadata.get(key) == value for key, value in where.items())