from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    nvidia_api_key: str = ""
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    chat_model: str = "meta/llama-3.1-8b-instruct"
    embedding_model: str = "nvidia/nv-embedqa-e5-v5"

    index_dir: str = "storage/faiss_index"
    docs_dir: str = "data/sample_docs"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    max_critic_revisions: int = 2
    temperature: float = 0.1

    def require_key(self) -> str:
        if not self.nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY not set. Get a free key at https://build.nvidia.com")
        return self.nvidia_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
