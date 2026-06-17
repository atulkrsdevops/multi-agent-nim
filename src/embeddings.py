from __future__ import annotations
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from .settings import get_settings


def get_embeddings() -> NVIDIAEmbeddings:
    s = get_settings()
    return NVIDIAEmbeddings(
        model=s.embedding_model,
        base_url=s.nim_base_url,
        api_key=s.require_key(),
        truncate="END",
    )
