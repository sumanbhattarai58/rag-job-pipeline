"""FastAPI application entrypoint."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.llm.hf_llm import HFLLM
from app.retrieval.hybrid_retriever import HybridRetriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retriever = HybridRetriever()
    app.state.llm = HFLLM()
    yield


app = FastAPI(title="LF Jobs RAG API", lifespan=lifespan)
app.include_router(router, prefix="/api")