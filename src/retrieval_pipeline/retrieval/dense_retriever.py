from __future__ import annotations

from retrieval_pipeline.indexing.faiss_index import FaissIndex
from retrieval_pipeline.models.embedding_model import EmbeddingModel
from retrieval_pipeline.schemas import SearchHit


class DenseRetriever:
    def __init__(self, embedder: EmbeddingModel, faiss_index: FaissIndex) -> None:
        self.embedder = embedder
        self.faiss_index = faiss_index

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        query_embedding = self.embedder.encode_query(query)
        return self.faiss_index.search(query_embedding=query_embedding, top_k=top_k)
