from __future__ import annotations
import pathlib
from langchain_community.vectorstores import FAISS
from .embeddings import get_embeddings
from .settings import get_settings


def get_retriever():
    s = get_settings()
    if not pathlib.Path(s.index_dir).exists():
        raise FileNotFoundError(f"No index at {s.index_dir}. Run `python -m src.ingest` first.")
    store = FAISS.load_local(s.index_dir, get_embeddings(), allow_dangerous_deserialization=True)
    return store.as_retriever(search_kwargs={"k": s.top_k})
