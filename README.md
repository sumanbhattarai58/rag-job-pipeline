# LF Jobs RAG Pipeline

A Retrieval-Augmented Generation pipeline over the LF Jobs dataset (1,000 job listings), with hybrid (vector + keyword) search and a FastAPI query endpoint.

## Setup

1. **Clone and create a virtual environment**
   ```
   git clone <your-repo-url>
   cd Job Data Retrieval
   python -m venv venv            #python 3.11 recommended
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # macOS/Linux
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```
   copy .env.example .env         # Windows
   cp .env.example .env           # macOS/Linux
   ```
   Edit `.env` and set `HF_API_TOKEN` to a Hugging Face token with Inference API access.

4. **Add the dataset**
   Place the LF Jobs CSV at `data/lf_jobs.csv`.

5. **Build the vector index** (one-time, run again after any dataset or chunking config change)
   ```
   python scripts/build_index.py
   ```

6. **Run the API**
   ```
   uvicorn app.api.main:app --reload
   ```
   Swagger UI: http://127.0.0.1:8000/docs

## Project structure
```
app/
  config.py            Centralized settings
  ingestion/            CSV loading, cleaning, chunking
  embeddings/           Local sentence-transformers embedder
  vectorstore/          Chroma wrapper
  retrieval/             Vector, BM25, and hybrid retrievers
  llm/                   Hugging Face LLM wrapper and prompts
  api/                    FastAPI app, routes, schemas
scripts/
  build_index.py         One-time indexing script
docs/
  architecture.md         Full documentation report
data/
  lf_jobs.csv              Source dataset
```

## Example request
```json
POST /api/query
{
  "query": "senior data scientist jobs with Python experience",
  "top_n": 5,
  "job_category": "Data and Analytics",
  "job_level": "Senior Level"
}
```