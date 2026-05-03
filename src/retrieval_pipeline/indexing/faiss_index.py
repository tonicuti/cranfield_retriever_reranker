from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from retrieval_pipeline.schemas import Document, SearchHit


class FaissIndex:
    def __init__(self, metric: str = "cosine") -> None:
        self.metric = metric
        self.index: faiss.Index | None = None
        self.documents: list[Document] = []

    def build(self, embeddings: np.ndarray, documents: list[Document]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix.")

        if len(documents) != len(embeddings):
            raise ValueError("Document count must match embedding count.")

        dimension = embeddings.shape[1]
        if self.metric == "cosine":
            self.index = faiss.IndexFlatIP(dimension)
        elif self.metric == "l2":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported FAISS metric: {self.metric}")

        self.index.add(embeddings)
        self.documents = documents

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchHit]:
        if self.index is None:
            raise RuntimeError("FAISS index has not been built or loaded.")

        scores, indices = self.index.search(query_embedding, top_k)
        hits: list[SearchHit] = []

        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            doc = self.documents[int(idx)]
            hits.append(SearchHit(doc_id=doc.doc_id, score=float(score), text=doc.text))

        return hits

    def save(self, index_dir: Path) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save an empty index.")

        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "index.faiss"))

        metadata = [
            {"doc_id": document.doc_id, "text": document.text}
            for document in self.documents
        ]
        with (index_dir / "documents.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False)

    @classmethod
    def load(cls, index_dir: Path, metric: str = "cosine") -> "FaissIndex":
        instance = cls(metric=metric)
        instance.index = faiss.read_index(str(index_dir / "index.faiss"))

        with (index_dir / "documents.json").open("r", encoding="utf-8") as file:
            raw_documents = json.load(file)

        instance.documents = [
            Document(doc_id=item["doc_id"], text=item["text"])
            for item in raw_documents
        ]
        return instance
