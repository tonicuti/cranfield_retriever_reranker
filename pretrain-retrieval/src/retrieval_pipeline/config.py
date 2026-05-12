from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(slots=True)
class PathsConfig:
    docs_dir: str
    queries_file: str
    qrels_dir: str
    index_dir: str
    run_dir: str

    @property
    def docs_dir_path(self) -> Path:
        return _resolve_path(self.docs_dir)

    @property
    def queries_file_path(self) -> Path:
        return _resolve_path(self.queries_file)

    @property
    def qrels_dir_path(self) -> Path:
        return _resolve_path(self.qrels_dir)

    @property
    def index_dir_path(self) -> Path:
        return _resolve_path(self.index_dir)

    @property
    def run_dir_path(self) -> Path:
        return _resolve_path(self.run_dir)


@dataclass(slots=True)
class IndexingConfig:
    embedder_model: str
    embedding_device: str = "cpu"
    normalize_embeddings: bool = True
    batch_size: int = 64
    faiss_metric: str = "cosine"
    top_k: int = 100


@dataclass(slots=True)
class RerankingConfig:
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    device: str = "cpu"
    top_k_candidates: int = 100
    top_k_final: int = 10


@dataclass(slots=True)
class Settings:
    paths: PathsConfig
    indexing: IndexingConfig
    reranking: RerankingConfig


def load_settings(config_path: str | Path) -> Settings:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    return Settings(
        paths=PathsConfig(**raw["paths"]),
        indexing=IndexingConfig(**raw["indexing"]),
        reranking=RerankingConfig(**raw["reranking"]),
    )
