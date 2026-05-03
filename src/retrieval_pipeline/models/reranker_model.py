from __future__ import annotations

from sentence_transformers.cross_encoder import CrossEncoder

from retrieval_pipeline.schemas import SearchHit


class RerankerModel:
    def __init__(self, model_name_or_path: str, device: str = "cpu") -> None:
        self.model_name_or_path = model_name_or_path
        self.model = CrossEncoder(model_name_or_path, device=device)

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []

        pairs = [[query, hit.text] for hit in hits]
        scores = self.model.predict(pairs)

        reranked = [
            SearchHit(doc_id=hit.doc_id, score=float(score), text=hit.text)
            for hit, score in zip(hits, scores, strict=True)
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]
