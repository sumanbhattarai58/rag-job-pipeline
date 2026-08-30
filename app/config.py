from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    hf_api_token: str = ""
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # hf_embedding_provider: str = "hf-inference"
    hf_llm_model: str = "deepseek-ai/DeepSeek-V4-Flash"
    hf_llm_provider: str = "novita"

    data_path: str = "data/lf_jobs.csv"

    chroma_persist_dir: str = "data/chroma_store"
    chroma_collection_name: str = "lf_jobs"

    top_n_default: int = 5
    vector_weight: float = 0.6
    bm25_weight: float = 0.4

    chunk_size_tokens: int = 300
    chunk_overlap_tokens: int = 50


settings = Settings()