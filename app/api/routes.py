"""API routes."""
from fastapi import APIRouter, Request
from app.api.schemas import JobResult, QueryRequest, QueryResponse
from app.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.postprocess import dedupe_by_job

EXCERPT_LENGTH = 300
CANDIDATE_MULTIPLIER = 3

JOB_CATEGORIES = [
    "Data and Analytics",
    "Software Engineering",
    "Design and UX",
    "Sales",
    "Project Management",
    "Advertising and Marketing",
    "General",
]
JOB_LEVELS = ["Senior Level", "Mid Level", "Entry Level", "Internship"]
_CATEGORY_LOOKUP = {v.lower(): v for v in JOB_CATEGORIES}
_LEVEL_LOOKUP = {v.lower(): v for v in JOB_LEVELS}

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, http_request: Request) -> QueryResponse:
    retriever = http_request.app.state.retriever
    llm = http_request.app.state.llm

    where = _build_filter(request)
    hits = retriever.search(request.query, top_n=request.top_n * CANDIDATE_MULTIPLIER, where=where)
    deduped = dedupe_by_job(hits)[:request.top_n]

    answer = llm.generate(SYSTEM_PROMPT, build_user_prompt(request.query, deduped))

    results = [
        JobResult(
            job_id=hit["metadata"]["job_id"],
            job_title=hit["metadata"]["job_title"],
            company_name=hit["metadata"]["company_name"],
            job_category=hit["metadata"]["job_category"],
            job_level=hit["metadata"]["job_level"],
            job_location=hit["metadata"]["job_location"],
            publication_date=hit["metadata"]["publication_date"],
            score=hit["score"],
            matched_excerpt=hit["text"][:EXCERPT_LENGTH],
        )
        for hit in deduped
    ]

    return QueryResponse(query=request.query, results=results, answer=answer)


def _build_filter(request: QueryRequest) -> dict | None:
    filters = {}
    if request.job_category:
        filters["job_category"] = _CATEGORY_LOOKUP.get(request.job_category.lower(), request.job_category)
    if request.job_level:
        filters["job_level"] = _LEVEL_LOOKUP.get(request.job_level.lower(), request.job_level)
    return filters or None