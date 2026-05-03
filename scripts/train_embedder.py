from __future__ import annotations

import random
import sys
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer, losses
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
    output = "artifacts/models/cranfield_embedder"
    epochs = 3
    batch_size = 16
    warmup_ratio = 0.1
    seed = 42


def build_training_examples(config_path: str) -> list[InputExample]:
    settings = load_settings(config_path)
    queries = load_queries(settings.paths.queries_file_path)
    qrels = load_qrels(settings.paths.qrels_dir_path)
    documents = {
        document.doc_id: document.text
        for document in load_documents(settings.paths.docs_dir_path)
    }

    examples: list[InputExample] = []
    for query_id, doc_scores in qrels.items():
        query_text = queries.get(query_id)
        if query_text is None:
            continue

        for doc_id, grade in doc_scores.items():
            if grade <= 0 or doc_id not in documents:
                continue
            examples.append(InputExample(texts=[query_text, documents[doc_id]]))

    return examples


def main() -> None:
    random.seed(CFG.seed)

    settings = load_settings(CFG.config)
    base_model = CFG.base_model or settings.indexing.embedder_model

    train_examples = build_training_examples(CFG.config)
    if not train_examples:
        raise RuntimeError("No positive query-document pairs were built from qrels.")

    random.shuffle(train_examples)
    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=CFG.batch_size,
    )

    model = SentenceTransformer(
        base_model,
        device=settings.indexing.embedding_device,
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = int(len(train_loader) * CFG.epochs * CFG.warmup_ratio)

    print(f"Base embedder: {base_model}")
    print(f"Training pairs: {len(train_examples)}")
    print(f"Output: {CFG.output}")

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=CFG.epochs,
        warmup_steps=warmup_steps,
        output_path=CFG.output,
        show_progress_bar=True,
    )


if __name__ == "__main__":
    main()
