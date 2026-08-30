"""One-time script: load, clean, chunk, embed, and store the dataset into Chroma."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from app.config import settings
from app.embeddings.hf_embedder import HFEmbedder
from app.ingestion.loader import load_jobs
from app.ingestion.pipeline import build_records, record_to_metadata
from app.vectorstore.chroma_store import ChromaStore

BATCH_SIZE = 32


def main():
    df = load_jobs(settings.data_path)
    records = build_records(df, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    print(f"Built {len(records)} chunks from {len(df)} jobs")

    embedder = HFEmbedder()
    store = ChromaStore()

    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Indexing"):
        batch = records[i:i + BATCH_SIZE]
        embeddings = embedder.embed([r.text for r in batch])
        store.add(
            ids=[r.chunk_id for r in batch],
            embeddings=embeddings,
            documents=[r.text for r in batch],
            metadatas=[record_to_metadata(r) for r in batch],
        )

    print(f"Indexed {store.count()} chunks into collection '{settings.chroma_collection_name}'")


if __name__ == "__main__":
    main()