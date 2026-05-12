from __future__ import annotations

from retrieval_pipeline.config import Settings
from retrieval_pipeline.indexing.faiss_index import FaissIndex
from retrieval_pipeline.models.embedding_model import EmbeddingModel
from retrieval_pipeline.models.reranker_model import RerankerModel
from retrieval_pipeline.retrieval.dense_retriever import DenseRetriever
from retrieval_pipeline.schemas import SearchHit


class RetrievalPipeline:
    def __init__(
        self,
        retriever: DenseRetriever,
        reranker: RerankerModel | None,
        settings: Settings,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.settings = settings

    @classmethod
    def from_settings(cls, settings: Settings) -> "RetrievalPipeline":
        embedder = EmbeddingModel(
            model_name_or_path=settings.indexing.embedder_model,
            device=settings.indexing.embedding_device,
            normalize_embeddings=settings.indexing.normalize_embeddings,
        )
        faiss_index = FaissIndex.load(
            index_dir=settings.paths.index_dir_path,
            metric=settings.indexing.faiss_metric,
        )
        retriever = DenseRetriever(embedder=embedder, faiss_index=faiss_index)

        reranker = None
        if settings.reranking.enabled:
            reranker = RerankerModel(
                model_name_or_path=settings.reranking.model,
                device=settings.reranking.device,
            )

        return cls(retriever=retriever, reranker=reranker, settings=settings)

    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        requested_top_k = (
            top_k if top_k is not None else self.settings.reranking.top_k_final
        )
        candidate_k = max(self.settings.reranking.top_k_candidates, requested_top_k)
        hits = self.retriever.search(
            query=query,
            top_k=candidate_k,
        )

        if self.reranker is None:
            final_top_k = (
                top_k if top_k is not None else self.settings.indexing.top_k
            )
            return hits[:final_top_k]

        return self.reranker.rerank(
            query=query,
            hits=hits,
            top_k=requested_top_k,
        )
