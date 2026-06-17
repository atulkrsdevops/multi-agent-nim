from __future__ import annotations
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from .settings import get_settings


def get_chat_model(temperature: float | None = None) -> ChatNVIDIA:
    s = get_settings()
    return ChatNVIDIA(
        model=s.chat_model,
        base_url=s.nim_base_url,
        api_key=s.require_key(),
        temperature=s.temperature if temperature is None else temperature,
        max_tokens=1024,
    )
