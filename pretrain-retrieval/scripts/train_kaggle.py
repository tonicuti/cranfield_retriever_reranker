from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader

PROJECT_ROOT = Path.cwd()


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


@dataclass(slots=True)
class Document:
    doc_id: str
    text: str


def load_settings(config: str | Path | dict) -> Settings:
    if isinstance(config, dict):
        raw = config
    else:
        path = Path(config)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        with path.open("r", encoding="utf-8") as file:
            raw = json.load(file)

    return Settings(
        paths=PathsConfig(**raw["paths"]),
        indexing=IndexingConfig(**raw["indexing"]),
        reranking=RerankingConfig(**raw["reranking"]),
    )


def load_documents(docs_dir: Path) -> list[Document]:
    documents: list[Document] = []

    for path in sorted(docs_dir.glob("*.txt"), key=lambda item: int(item.stem)):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        documents.append(Document(doc_id=path.stem, text=text))

    return documents


def load_queries(queries_file: Path) -> dict[str, str]:
    queries: dict[str, str] = {}

    with queries_file.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            query_id, text = line.split("\t", maxsplit=1)
            queries[query_id] = text.strip()

    return queries


def load_qrels(qrels_dir: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}

    for path in sorted(qrels_dir.glob("*.txt"), key=lambda item: int(item.stem)):
        query_id = path.stem
        qrels[query_id] = {}

        with path.open("r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                _, doc_id, grade = parts
                qrels[query_id][doc_id] = int(grade)

    return qrels


class CFG:
    config = {
        "paths": {
            "docs_dir": "/kaggle/input/datasets/toni0710/cranfield/cranfield-dataset/DOCS",
            "queries_file": "/kaggle/input/datasets/toni0710/cranfield/cranfield-dataset/query.txt",
            "qrels_dir": "/kaggle/input/datasets/toni0710/cranfield/cranfield-dataset/REL",
            "index_dir": "/kaggle/working/indexes",
            "run_dir": "/kaggle/working/runs",
        },
        "indexing": {
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_device": "cuda",
            "normalize_embeddings": True,
            "batch_size": 64,
            "faiss_metric": "cosine",
            "top_k": 100,
        },
        "reranking": {
            "enabled": True,
            "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "device": "cuda",
            "top_k_candidates": 100,
            "top_k_final": 10,
        },
    }

    train_embedder = True
    train_reranker = True
    train_ratio = 0.8
    seed = 42

    embedder_base_model: str | None = None
    embedder_output = "/kaggle/working/cranfield_embedder"
    embedder_epochs = 3
    embedder_batch_size = 16
    embedder_warmup_ratio = 0.1

    reranker_base_model: str | None = None
    reranker_output = "/kaggle/working/cranfield_reranker"
    reranker_epochs = 3
    reranker_batch_size = 8
    reranker_max_length = 512
    reranker_document_word_limit = 256
    negatives_per_positive = 3
    reranker_warmup_ratio = 0.1
    zip_outputs = True


def split_query_ids(qrels: dict[str, dict[str, int]]) -> tuple[set[str], set[str]]:
    query_ids = sorted(qrels)
    random.shuffle(query_ids)

    split_index = int(len(query_ids) * CFG.train_ratio)
    train_query_ids = set(query_ids[:split_index])
    test_query_ids = set(query_ids[split_index:])
    return train_query_ids, test_query_ids


def load_training_data() -> tuple[
    Settings,
    dict[str, str],
    dict[str, dict[str, int]],
    dict[str, str],
    set[str],
    set[str],
]:
    settings = load_settings(CFG.config)
    queries = load_queries(settings.paths.queries_file_path)
    qrels = load_qrels(settings.paths.qrels_dir_path)
    train_query_ids, test_query_ids = split_query_ids(qrels)
    documents = {
        document.doc_id: document.text
        for document in load_documents(settings.paths.docs_dir_path)
    }
    return settings, queries, qrels, documents, train_query_ids, test_query_ids


def build_embedder_examples(
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    documents: dict[str, str],
    train_query_ids: set[str],
) -> list[InputExample]:
    examples: list[InputExample] = []
    for query_id, doc_scores in qrels.items():
        if query_id not in train_query_ids:
            continue

        query_text = queries.get(query_id)
        if query_text is None:
            continue

        for doc_id, grade in doc_scores.items():
            if grade <= 0 or doc_id not in documents:
                continue
            examples.append(InputExample(texts=[query_text, documents[doc_id]]))

    return examples


def build_reranker_examples(
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    documents: dict[str, str],
    train_query_ids: set[str],
) -> list[InputExample]:
    all_doc_ids = list(documents)
    examples: list[InputExample] = []

    for query_id, doc_scores in qrels.items():
        if query_id not in train_query_ids:
            continue

        query_text = queries.get(query_id)
        if query_text is None:
            continue

        relevant_doc_ids = {
            doc_id
            for doc_id, grade in doc_scores.items()
            if grade > 0 and doc_id in documents
        }
        if not relevant_doc_ids:
            continue

        for doc_id in relevant_doc_ids:
            examples.append(
                InputExample(
                    texts=[
                        query_text,
                        truncate_document_for_reranker(documents[doc_id]),
                    ],
                    label=1.0,
                )
            )

        negative_pool = [
            doc_id for doc_id in all_doc_ids if doc_id not in relevant_doc_ids
        ]
        negative_count = min(
            len(relevant_doc_ids) * CFG.negatives_per_positive,
            len(negative_pool),
        )

        for doc_id in random.sample(negative_pool, k=negative_count):
            examples.append(
                InputExample(
                    texts=[
                        query_text,
                        truncate_document_for_reranker(documents[doc_id]),
                    ],
                    label=0.0,
                )
            )

    return examples


def truncate_document_for_reranker(text: str) -> str:
    words = text.split()
    if len(words) <= CFG.reranker_document_word_limit:
        return text
    return " ".join(words[: CFG.reranker_document_word_limit])


def train_embedder(
    settings: Settings,
    train_examples: list[InputExample],
    train_query_count: int,
    held_out_query_count: int,
) -> None:
    if not train_examples:
        raise RuntimeError("No positive query-document pairs were built for embedder.")

    random.shuffle(train_examples)
    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=CFG.embedder_batch_size,
    )

    base_model = CFG.embedder_base_model or settings.indexing.embedder_model
    model = SentenceTransformer(
        base_model,
        device=settings.indexing.embedding_device,
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = int(
        len(train_loader) * CFG.embedder_epochs * CFG.embedder_warmup_ratio
    )

    print("\n=== Train embedder ===")
    print(f"Base embedder: {base_model}")
    print(f"Train query count: {train_query_count}")
    print(f"Held-out query count: {held_out_query_count}")
    print(f"Training pairs: {len(train_examples)}")
    print(f"Output: {CFG.embedder_output}")

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=CFG.embedder_epochs,
        warmup_steps=warmup_steps,
        output_path=CFG.embedder_output,
        show_progress_bar=True,
    )
    model.save(CFG.embedder_output)
    maybe_zip_output(CFG.embedder_output)


def train_reranker(
    settings: Settings,
    train_examples: list[InputExample],
    train_query_count: int,
    held_out_query_count: int,
) -> None:
    if not train_examples:
        raise RuntimeError("No reranker training pairs were built from qrels.")

    random.shuffle(train_examples)
    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=CFG.reranker_batch_size,
    )

    base_model = CFG.reranker_base_model or settings.reranking.model
    model = CrossEncoder(
        base_model,
        num_labels=1,
        max_length=CFG.reranker_max_length,
        device=settings.reranking.device,
    )
    model.max_length = CFG.reranker_max_length
    model.tokenizer.model_max_length = CFG.reranker_max_length
    warmup_steps = int(
        len(train_loader) * CFG.reranker_epochs * CFG.reranker_warmup_ratio
    )

    print("\n=== Train reranker ===")
    print(f"Base reranker: {base_model}")
    print(f"Train query count: {train_query_count}")
    print(f"Held-out query count: {held_out_query_count}")
    print(f"Training pairs: {len(train_examples)}")
    print(f"Max sequence length: {CFG.reranker_max_length}")
    print(f"Document word limit: {CFG.reranker_document_word_limit}")
    print(f"Output: {CFG.reranker_output}")

    model.fit(
        train_dataloader=train_loader,
        epochs=CFG.reranker_epochs,
        warmup_steps=warmup_steps,
        output_path=CFG.reranker_output,
        show_progress_bar=True,
    )
    model.save(CFG.reranker_output)
    maybe_zip_output(CFG.reranker_output)


def maybe_zip_output(output_dir: str) -> None:
    output_path = Path(output_dir)
    if not output_path.exists():
        raise RuntimeError(f"Expected model output directory was not created: {output_dir}")

    print(f"Saved model directory: {output_path}")

    if not CFG.zip_outputs:
        return

    zip_base = str(output_path)
    zip_path = shutil.make_archive(zip_base, "zip", root_dir=output_path)
    print(f"Saved zip archive: {zip_path}")


def main() -> None:
    random.seed(CFG.seed)
    settings, queries, qrels, documents, train_query_ids, test_query_ids = (
        load_training_data()
    )

    print(f"Total query count: {len(qrels)}")
    print(f"Train query count: {len(train_query_ids)}")
    print(f"Held-out query count: {len(test_query_ids)}")

    if CFG.train_embedder:
        embedder_examples = build_embedder_examples(
            queries=queries,
            qrels=qrels,
            documents=documents,
            train_query_ids=train_query_ids,
        )
        train_embedder(
            settings=settings,
            train_examples=embedder_examples,
            train_query_count=len(train_query_ids),
            held_out_query_count=len(test_query_ids),
        )

    if CFG.train_reranker:
        reranker_examples = build_reranker_examples(
            queries=queries,
            qrels=qrels,
            documents=documents,
            train_query_ids=train_query_ids,
        )
        train_reranker(
            settings=settings,
            train_examples=reranker_examples,
            train_query_count=len(train_query_ids),
            held_out_query_count=len(test_query_ids),
        )


if __name__ == "__main__":
    main()
