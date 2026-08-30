# """Embedding generation via the Hugging Face Inference API."""
# import time

# import numpy as np
# from huggingface_hub import InferenceClient

# from app.config import settings

# MAX_RETRIES = 3
# RETRY_DELAY_SECONDS = 2


# class HFEmbedder:
#     """Generates sentence embeddings via the Hugging Face Inference API."""

#     def __init__(self, model: str | None = None, token: str | None = None, provider: str | None = None):
#         self.client = InferenceClient(
#             model=model or settings.hf_embedding_model,
#             token=token or settings.hf_api_token,
#             provider=provider or settings.hf_embedding_provider,
#         )

#     def embed(self, texts: list[str]) -> list[list[float]]:
#         return [self._embed_one(text) for text in texts]

#     def _embed_one(self, text: str) -> list[float]:
#         vector = None
#         for attempt in range(MAX_RETRIES):
#             try:
#                 vector = np.array(self.client.feature_extraction(text))
#                 break
#             except Exception:
#                 if attempt == MAX_RETRIES - 1:
#                     raise
#                 time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
#         if vector.ndim == 2:
#             vector = vector.mean(axis=0)
#         return vector.tolist()


"""Local embedding generation using Sentence Transformers."""

from sentence_transformers import SentenceTransformer

from app.config import settings


class HFEmbedder:
    """Generates sentence embeddings locally."""

    def __init__(
        self,
        model: str | None = None,
        token: str | None = None,
        provider: str | None = None,
    ):
        self.model = SentenceTransformer(
            model or settings.hf_embedding_model
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()


