
from dataclasses import asdict, dataclass
import pandas as pd
from app.ingestion.chunker import chunk_job_description
from app.ingestion.cleaner import clean_field


@dataclass
class JobChunkRecord:
    chunk_id: str          # e.g. "LF0001-3" (job id + chunk index)
    job_id: str
    text: str
    section_label: str
    job_title: str
    company_name: str
    job_category: str
    job_level: str
    job_location: str
    publication_date: str  # ISO string, kept as string for vector-store metadata compatibility
    tags: str


def build_records(df: pd.DataFrame, chunk_size_tokens: int, overlap_tokens: int) -> list[JobChunkRecord]:
    records = []
    for _, row in df.iterrows():
        chunks = chunk_job_description(row["Job Description"], chunk_size_tokens, overlap_tokens)
        for chunk in chunks:
            records.append(JobChunkRecord(
                chunk_id=f"{row['ID']}-{chunk.chunk_index}",
                job_id=row["ID"],
                text=chunk.text,
                section_label=chunk.section_label,
                job_title=clean_field(row["Job Title"]),
                company_name=clean_field(row["Company Name"]),
                job_category=clean_field(row["Job Category"]),
                job_level=clean_field(row["Job Level"]),
                job_location=clean_field(row.get("Job Location")) or "Unknown",
                publication_date=str(row["Publication Date"]) if pd.notna(row["Publication Date"]) else "",
                tags=clean_field(row.get("Tags")),
            ))
    return records


def record_to_metadata(record: JobChunkRecord) -> dict:
    """Metadata dict for the vector store — everything except the embedded text itself."""
    d = asdict(record)
    d.pop("text")
    return d 