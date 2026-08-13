from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Iterable

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class SearchResult:
    text: str
    filename: str
    page: int
    score: float


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _chunk_words(text: str, chunk_size: int = 350, overlap: int = 60) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


class FinancialDocumentStore:
    """In-memory Chroma store backed by local sentence-transformer embeddings."""

    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.embedding_model_name = embedding_model
        self._model: SentenceTransformer | None = None
        self._client = chromadb.Client()
        self._collection = self._new_collection()

    def _new_collection(self):
        name = f"financial-statements-{hashlib.sha1(str(id(self)).encode()).hexdigest()[:12]}"
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection.name)
        except Exception:
            pass
        self._collection = self._new_collection()

    def ingest(self, uploaded_files: Iterable[Any]) -> dict[str, int]:
        self.reset()
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []
        file_count = 0
        page_count = 0

        for uploaded_file in uploaded_files:
            file_count += 1
            filename = uploaded_file.name
            file_bytes = uploaded_file.getvalue()
            reader = PdfReader(BytesIO(file_bytes))

            for page_number, page in enumerate(reader.pages, start=1):
                page_text = _normalise_text(page.extract_text() or "")
                if not page_text:
                    continue
                page_count += 1
                for chunk_number, chunk in enumerate(_chunk_words(page_text)):
                    document_id = hashlib.sha256(
                        f"{filename}:{page_number}:{chunk_number}:{chunk}".encode("utf-8")
                    ).hexdigest()
                    ids.append(document_id)
                    documents.append(chunk)
                    metadatas.append(
                        {
                            "filename": filename,
                            "page": page_number,
                            "chunk": chunk_number,
                        }
                    )

        if not documents:
            raise ValueError(
                "No readable text was found. Scanned PDFs need OCR before they can be searched."
            )

        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        return {
            "files": file_count,
            "pages": page_count,
            "chunks": len(documents),
        }

    def search(self, question: str, top_k: int = 6) -> list[SearchResult]:
        if self._collection.count() == 0:
            raise ValueError("No documents have been indexed yet.")

        query_embedding = self.model.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        result = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        matches: list[SearchResult] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            matches.append(
                SearchResult(
                    text=text,
                    filename=str(metadata["filename"]),
                    page=int(metadata["page"]),
                    score=max(0.0, 1.0 - float(distance)),
                )
            )
        return matches
