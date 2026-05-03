from __future__ import annotations

from retrieval_pipeline.config import Settings
from retrieval_pipeline.data.cranfield_loader import load_documents
from retrieval_pipeline.indexing.faiss_index import FaissIndex
from retrieval_pipeline.models.embedding_model import EmbeddingModel


def build_faiss_index(settings: Settings) -> None:
    documents = load_documents(settings.paths.docs_dir_path)

    embedder = EmbeddingModel(
        model_name_or_path=settings.indexing.embedder_model,
        device=settings.indexing.embedding_device,
        normalize_embeddings=settings.indexing.normalize_embeddings,
    )
    embeddings = embedder.encode_texts(
        [document.text for document in documents],
        batch_size=settings.indexing.batch_size,
    )

    faiss_index = FaissIndex(metric=settings.indexing.faiss_metric)
    faiss_index.build(embeddings=embeddings, documents=documents)
    faiss_index.save(settings.paths.index_dir_path)
