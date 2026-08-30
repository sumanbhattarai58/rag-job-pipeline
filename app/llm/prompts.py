"""Prompt construction for grounded, retrieval-based responses."""

SYSTEM_PROMPT = (
    "You are a job search assistant. Answer the user's query using only the "
    "job listings provided below. Be concise and specific: reference job "
    "titles and companies, and explain briefly why each listing matches. "
    "If none of the listings match well, say so directly instead of guessing."
)

EXCERPT_LENGTH = 400


def build_user_prompt(query: str, hits: list[dict]) -> str:
    listings = []
    for i, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]
        listings.append(
            f"[{i}] {metadata['job_title']} at {metadata['company_name']} "
            f"({metadata['job_level']}, {metadata['job_location']}, "
            f"category: {metadata['job_category']})\n"
            f"Excerpt: {hit['text'][:EXCERPT_LENGTH]}"
        )
    context = "\n\n".join(listings)
    return (
        f"User query: {query}\n\n"
        f"Retrieved job listings:\n{context}\n\n"
        f"Respond to the user's query using these listings."
    )