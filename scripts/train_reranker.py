from __future__ import annotations

import random
import sys
from pathlib import Path

from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval_pipeline.config import load_settings  # noqa: E402
from retrieval_pipeline.data.cranfield_loader import (  # noqa: E402
    load_documents,
    load_qrels,
    load_queries,
)


class CFG:
    config = "configs/base.json"
    base_model: str | None = None
    output = "artifacts/models/cranfield_reranker"
    epochs = 3
    batch_size = 8
    negatives_per_positive = 3
    warmup_ratio = 0.1
    seed = 42


def build_training_examples(
    config_path: str,
    negatives_per_positive: int,
) -> list[InputExample]:
    settings = load_settings(config_path)
    queries = load_queries(settings.paths.queries_file_path)
    qrels = load_qrels(settings.paths.qrels_dir_path)
    documents = {
        document.doc_id: document.text
        for document in load_documents(settings.paths.docs_dir_path)
    }
    all_doc_ids = list(documents)

    examples: list[InputExample] = []
    for query_id, doc_scores in qrels.items():
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
                    texts=[query_text, documents[doc_id]],
                    label=1.0,
                )
            )

        negative_pool = [
            doc_id for doc_id in all_doc_ids if doc_id not in relevant_doc_ids
        ]
        negative_count = min(
            len(relevant_doc_ids) * negatives_per_positive,
            len(negative_pool),
        )

        for doc_id in random.sample(negative_pool, k=negative_count):
            examples.append(
                InputExample(
                    texts=[query_text, documents[doc_id]],
                    label=0.0,
                )
            )

    return examples


def main() -> None:
    random.seed(CFG.seed)

    settings = load_settings(CFG.config)
    base_model = CFG.base_model or settings.reranking.model

    train_examples = build_training_examples(
        config_path=CFG.config,
        negatives_per_positive=CFG.negatives_per_positive,
    )
    if not train_examples:
        raise RuntimeError("No reranker training pairs were built from qrels.")

    random.shuffle(train_examples)
    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=CFG.batch_size,
    )

    model = CrossEncoder(
        base_model,
        num_labels=1,
        device=settings.reranking.device,
    )
    warmup_steps = int(len(train_loader) * CFG.epochs * CFG.warmup_ratio)

    print(f"Base reranker: {base_model}")
    print(f"Training pairs: {len(train_examples)}")
    print(f"Output: {CFG.output}")

    model.fit(
        train_dataloader=train_loader,
        epochs=CFG.epochs,
        warmup_steps=warmup_steps,
        output_path=CFG.output,
        show_progress_bar=True,
    )


if __name__ == "__main__":
    main()
