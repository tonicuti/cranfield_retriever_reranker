from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model_name_or_path, device=device)

    def encode_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )
        return embeddings.astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        embeddings = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")
