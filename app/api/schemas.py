"""Request and response schemas for the API."""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_n: int = Field(default=5, ge=1, le=20)
    job_category: str | None = None
    job_level: str | None = None


class JobResult(BaseModel):
    job_id: str
    job_title: str
    company_name: str
    job_category: str
    job_level: str
    job_location: str
    publication_date: str
    score: float
    matched_excerpt: str


class QueryResponse(BaseModel):
    query: str
    results: list[JobResult]
    answer: str