# LF Jobs RAG Pipeline : Documentation

## 1. High-Level Architecture and Engineering Decisions

### Pipeline overview
```
CSV (1,000 jobs)
  -> Loader (validates columns, parses dates)
  -> Cleaner (fixes encoding, strips HTML)
  -> Chunker (section-aware + fixed-size fallback)
  -> Records (chunk text + full job metadata)
  -> Embedder (sentence-transformers, local)
  -> Chroma (vector store, cosine similarity)

Query -> Hybrid Retriever (vector search + BM25, weighted merge)
      -> Dedup (best chunk per job)
      -> Prompt builder -> LLM (Hugging Face Inference API)
      -> JSON response (matched jobs + grounded answer)
```

### Decision: Preprocessing (Fixing broken text before reading HTML)
The raw job descriptions were stored in HTML format, and many had corrupted text (like â€™ instead of an apostrophe) caused by encoding glitches upstream. We used a tool called ftfy to fix this text before removing the HTML tags so that garbled words wouldn't ruin our search embeddings or LLM responses.

We used BeautifulSoup instead of basic regular expressions (regex) to strip out the HTML tags. Regex can easily break on bad or nested HTML, and several listings had broken structures (like tables that collapsed into a wall of unreadable text).

### Decision: Hybrid section-aware chunking
The job descriptions averaged 732 words (and went up to 2,122 words). Turning an entire job listing into a single chunk was a bad idea because search results would be too vague. We evaluated two options:

- **Fixed-size windows only:** Simple and reliable, but it cuts text blindly without respecting natural boundaries.

- **Section-aware chunking:** Splits text by bold titles or headings (like `<b>Responsibilities:</b>`) to keep sections like "Skills" or "Requirements" intact.

We built a hybrid system: it first attempts to split the text by sections (reading every text node line-by-line and grouping it under the nearest header). If it finds fewer than 2 sections, it falls back to fixed-size chunks (300 tokens with a 50-token overlap). If a single section is too long, it breaks that section down further into smaller windows.

This ensures unstructured listings still get chunked properly, while structured listings get neatly labeled sections. Testing on 1,000 listings (which created 6,145 chunks) helped us catch and fix two bugs:

- Content placed after the final header was being accidentally deleted (fixed by reading every single text node, not just `<p>` and `<li>` tags).

- A single bold colon (`<b>` : `</b>`) used as a divider was misidentified as an empty heading title.

### Decision: Embeddings (Running sentence-transformers locally)
**Originally planned**: embeddings via the Hugging Face Inference API (hosted, no local download).   
**What we shipped**: local embeddings via the `sentence-transformers` library (`all-MiniLM-L6-v2`).

This was a required change due to API failures. During development, Hugging Face’s free API repeatedly rejected our chosen models (like BAAI/bge-small-en-v1.5) with an error stating the model wasn't supported on the free serverless tier. Running the embedding model locally removed these external API errors completely, with the only tradeoff being a one-time model download and small local memory usage.

### Decision: Vector store (Chroma, cosine similarity)
We chose Chroma because it is simple, file-based, runs locally, and requires no extra infrastructure, making it a perfect fit for 6,145 chunks.
We specifically set the search metric to cosine similarity (hnsw:space: cosine) instead of Chroma’s default (squared L2 distance). Cosine similarity is the standard metric for sentence-embedding models, making the match scores much easier to understand and evaluate.

### Decision: Hybrid retrieval (vector + BM25)
Job listings depend heavily on exact keywords (like job titles, company names, or specific tools) that vector searches can sometimes miss or dilute.  
To solve this, we run a keyword search tool (BM25) alongside our vector search:  
- Both searches pull a larger pool of candidates than needed (4x the requested amount).  
- The score from each method is scaled to a standard 0 to 1 range.
- The scores are combined using a weighted mix (60% vector search, 40% keyword search).

This approach gives better keyword accuracy without needing a costly cross-encoder reranker model, keeping latency low and avoiding extra API calls.

### Decision: Removing duplicate jobs from results
Because search happens at the individual chunk level, a single job listing could have multiple high-scoring chunks, filling the top results with the same job.

To fix this, we fetch a larger list of candidate chunks, keep only the single highest-scoring chunk per job_id, and then trim the list down to the final count requested by the user.

### Decision: LLM Setup (Choosing a specific provider, streaming, and retries)
**Original Plan:** Use provider="auto" to let Hugging Face route requests to whichever server was free.

**What We Shipped:** Locked in a specific provider (novita, running deepseek-ai/DeepSeek-V4-Flash), checking its live availability using huggingface_hub tools.

provider="auto" kept failing with errors claiming no providers were available, even for models enabled on the account. Automatically routing requests was unreliable, but specifying the provider directly worked every time.

Additionally, "cold" models (models waking up from inactivity) often timed out with a 504 Gateway Time-out. To prevent the connection from timing out:

- We enabled response streaming (stream=True) so tokens start returning right away.  
- We added an automatic retry logic, which resolves the issue since cold-start models wake up on the second attempt.

### Decision: API Architecture (FastAPI with single-instance setup)
Our main retriever class (which builds the search index over all job data) and our LLM client are initialized once when the FastAPI app starts up (using FastAPI's lifespan context). They are stored in memory (app.state) and reused for every incoming request. Rebuilding the search index on every API call would be extremely slow and wasteful.

### Decision: Case-insensitive category/level filtering
When users filter by job category or seniority level (like "sales"), the API automatically converts their text to match the exact casing stored in the database (e.g., "Sales"). This ensures search queries work smoothly even if the user types in lowercase or mixed case.

---

## 2. Setup and Installation Instructions

See `README.md` in the repository root for the full step-by-step setup. Summary:
1. Clone repo, create a virtual environment, `pip install -r requirements.txt`
2. Copy `.env.example` to `.env`, add a Hugging Face token
3. Place the dataset CSV at `data/lf_jobs.csv`
4. Run `python scripts/build_index.py` once to build the vector index
5. Run `uvicorn app.api.main:app --reload`
6. Access Swagger UI at `http://127.0.0.1:8000/docs`

---

## 3. Example Usage

**Request:**
```json
POST /api/query
{
  "query": "customer relationship and sales experience",
  "top_n": 5,
  "job_category": "Sales",
  "job_level": "Senior Level"
}
```

**Response (abridged):**
```json
{
  "query": "customer relationship and sales experience",
  "results": [
    {
      "job_id": "LF0537",
      "job_title": "Territory Manager (outside sales) - Phoenix West",
      "company_name": "US Foods",
      "job_category": "Sales",
      "job_level": "Senior Level",
      "job_location": "Phoenix, AZ",
      "score": 0.88,
      "matched_excerpt": "1+ year of sales experience preferred..."
    }
  ],
  "answer": "All five retrieved listings match your query for customer relationship and sales experience. Here's a breakdown: [1] Territory Manager (outside sales) - Phoenix West at US Foods - requires sales experience, relationship building..."
}
```

**Unfiltered query:**
```json
{
  "query": "senior data scientist jobs with Python experience",
  "top_n": 5
}
```

---

## 4. Assumptions Made During Development

- **Consistent Data Structure:** We assume the dataset’s 9 columns (ID, Job Category, Job Title, Company Name, Publication Date, Job Location, Job Level, Tags, Job Description) will always stay the same and keep their current names.

- **Tags Are Unreliable for Filtering:** Only 120 out of 1,000 listings (12%) have data in the Tags column. Because it is mostly empty, we keep it as general information but do not use it to filter search results or influence ranking.

- **Handling Missing Locations:** A small number of listings (9 out of 1,000) were missing a Job Location. Instead of deleting these jobs, we default their location to "Unknown".

- **Headings Use Standard HTML Tags:** Based on looking through the data, job sections (like "Skills" or "Responsibilities") usually use `<b>`, `<strong>`, or heading tags. If a listing doesn't follow this pattern, our system simply defaults to dividing the text into fixed-size chunks instead.

- **Embedding Model Consistency:** You must use the exact same embedding model to create the index and to process user search queries. Switching models later would make existing search vectors incompatible, requiring the entire database to be re-indexed from scratch.

- **Hardware Requirements:** We assume the computer running the app has enough basic CPU power and memory to handle a lightweight embedding model (all-MiniLM-L6-v2 is only ~90MB and runs easily on a standard CPU).

---

## 5. Drawbacks and Future Enhancements

**Known drawbacks:**
- **No Dedicated Reranking Model:** We considered using a second-stage reranker model (a cross-encoder) to fine-tune result order, but chose a simpler hybrid search (vectors + keyword search) instead. This keeps search speeds fast and avoids relying on another external service.

- **Free Hosted Models Can Be Unreliable:** Free cloud-hosted models on Hugging Face occasionally went offline, moved, or failed during testing. While we fixed this by running our embedding model locally, locking in specific providers, and adding automatic retries, relying on free external APIs still carries some risk for real-world production.

- **Keyword Index Rebuilds on Startup:** Every time the application starts up, the keyword search index (BM25) is rebuilt from scratch using the CSV file. This works fine for 1,000 jobs, but would slow down startup significantly for larger datasets.

- **No API Security:** The API currently lacks user login authentication and rate limits to prevent abuse.

- **No Automated Search Testing:** Search accuracy was tested manually by hand rather than measured automatically against a formal benchmark dataset.

- **API Results Don't Stream to Users:** Even though our internal model streams text token-by-token on the backend, the API waits for the entire answer to finish generating before sending the final response back to the user.

**Future enhancements:**
- Add a secondary cross-encoder reranking step to make the top search results even more accurate.
- Build an automatic backup system that can switch providers if the primary cloud model goes offline.
- Save the keyword search index (BM25) to a file alongside the vector database so it doesn't need to rebuild every time the app starts.
- Stream the AI's response directly to the user in real time so they don't have to wait for the complete answer to generate.
- Create a set of test queries with known answers to automatically measure and score search accuracy over time.