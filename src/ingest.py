from __future__ import annotations
import pathlib
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .embeddings import get_embeddings
from .settings import get_settings


def _load_documents(docs_dir: str) -> list[Document]:
    root = pathlib.Path(docs_dir)
    docs = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in {".md", ".txt"} and path.is_file():
            docs.append(Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": str(path.relative_to(root))},
            ))
    if not docs:
        raise RuntimeError(f"No .md/.txt files found under {docs_dir}")
    return docs


def build_index() -> None:
    s = get_settings()
    docs = _load_documents(s.docs_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} docs -> {len(chunks)} chunks. Embedding via NIM...")
    store = FAISS.from_documents(chunks, get_embeddings())
    pathlib.Path(s.index_dir).parent.mkdir(parents=True, exist_ok=True)
    store.save_local(s.index_dir)
    print(f"Saved FAISS index to {s.index_dir}")


if __name__ == "__main__":
    build_index()
