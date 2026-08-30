"""Post-process retrieval hits before they reach the LLM or the API response."""


def dedupe_by_job(hits: list[dict]) -> list[dict]:
    """Keep only the highest-scoring chunk per job, so the same listing
    doesn't appear multiple times when several of its chunks match."""
    best_by_job = {}
    for hit in hits:
        job_id = hit["metadata"]["job_id"]
        if job_id not in best_by_job or hit["score"] > best_by_job[job_id]["score"]:
            best_by_job[job_id] = hit
    return sorted(best_by_job.values(), key=lambda h: h["score"], reverse=True)