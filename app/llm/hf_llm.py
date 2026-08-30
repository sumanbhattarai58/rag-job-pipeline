"""LLM response generation via the Hugging Face Inference API."""
import time

from huggingface_hub import InferenceClient

from app.config import settings

MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 3


class HFLLM:
    def __init__(self, model: str | None = None, provider: str | None = None):
        self.model = model or settings.hf_llm_model
        self.client = InferenceClient(token=settings.hf_api_token, provider=provider or settings.hf_llm_provider)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for attempt in range(MAX_RETRIES):
            try:
                chunks = self.client.chat_completion(
                    model=self.model, messages=messages, max_tokens=512, temperature=0.3, stream=True,
                )
                parts = []
                for chunk in chunks:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        parts.append(delta)
                return "".join(parts)
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(RETRY_DELAY_SECONDS)