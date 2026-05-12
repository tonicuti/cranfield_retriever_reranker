from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_SRC = PROJECT_ROOT / "pretrain-retrieval" / "src"
HYBRID_SRC = PROJECT_ROOT / "hybrid-retrieval"

sys.path.insert(0, str(PRETRAIN_SRC))
sys.path.insert(0, str(HYBRID_SRC))

from retrieval_pipeline.config import load_settings
from retrieval_pipeline.data.cranfield_loader import load_qrels, load_queries
from retrieval_pipeline.evaluation.cranfield_eval import (
    _binary_relevant_docs,
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    split_query_ids,
)

from search_rrf import (
    build_bm25_retriever,
    build_finetuned_components,
    hybrid_search,
)


class HybridRrfPipeline:
    def __init__(
        self,
        *,
        bm25,
        finetuned_biencoder,
        cross_encoder,
        docs_by_id,
        bm25_k: int,
        finetuned_k: int,
        rrf_top_n: int,
        rrf_k: int,
    ) -> None:
        self.bm25 = bm25
        self.finetuned_biencoder = finetuned_biencoder
        self.cross_encoder = cross_encoder
        self.docs_by_id = docs_by_id
        self.bm25_k = bm25_k
        self.finetuned_k = finetuned_k
        self.rrf_top_n = rrf_top_n
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 10):
        return hybrid_search(
            query,
            bm25=self.bm25,
            finetuned_biencoder=self.finetuned_biencoder,
            cross_encoder=self.cross_encoder,
            docs_by_id=self.docs_by_id,
            bm25_k=self.bm25_k,
            finetuned_k=self.finetuned_k,
            rrf_top_n=self.rrf_top_n,
            final_k=top_k,
            rrf_k=self.rrf_k,
        )


def evaluate_hybrid(pipeline: HybridRrfPipeline, config_path: Path, top_k: int):
    settings = load_settings(config_path)

    queries = load_queries(settings.paths.queries_file_path)
    qrels = load_qrels(settings.paths.qrels_dir_path)
    _, test_query_ids = split_query_ids(qrels)

    precision_total = 0.0
    recall_total = 0.0
    map_total = 0.0
    ndcg_total = 0.0
    query_count = 0

    for query_id, query_text in queries.items():
        if query_id not in test_query_ids:
            continue

        graded_relevance = qrels.get(query_id, {})
        relevant = _binary_relevant_docs(qrels, query_id)
        if not relevant:
            continue

        results = pipeline.search(query_text, top_k=top_k)
        retrieved_ids = [hit.doc_id for hit in results]

        precision_total += precision_at_k(retrieved_ids, relevant, top_k)
        recall_total += recall_at_k(retrieved_ids, relevant, top_k)
        map_total += average_precision_at_k(retrieved_ids, relevant, top_k)
        ndcg_total += ndcg_at_k(retrieved_ids, graded_relevance, top_k)
        query_count += 1

    return {
        "query_count": float(query_count),
        "precision@k": precision_total / query_count,
        "recall@k": recall_total / query_count,
        "map@k": map_total / query_count,
        "ndcg@k": ndcg_total / query_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--documents-csv",
        type=Path,
        default=PROJECT_ROOT / "traditional-retrieval" / "data" / "documents.csv",
    )
    parser.add_argument(
        "--finetuned-config",
        type=Path,
        default=PROJECT_ROOT / "pretrain-retrieval" / "configs" / "finetuned.json",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bm25-k", type=int, default=100)
    parser.add_argument("--finetuned-k", type=int, default=100)
    parser.add_argument("--rrf-top-n", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()

    bm25, docs_by_id = build_bm25_retriever(args.documents_csv)
    finetuned = build_finetuned_components(args.finetuned_config)

    pipeline = HybridRrfPipeline(
        bm25=bm25,
        finetuned_biencoder=finetuned.biencoder,
        cross_encoder=finetuned.cross_encoder,
        docs_by_id=docs_by_id,
        bm25_k=args.bm25_k,
        finetuned_k=args.finetuned_k,
        rrf_top_n=args.rrf_top_n,
        rrf_k=args.rrf_k,
    )

    metrics = evaluate_hybrid(
        pipeline=pipeline,
        config_path=args.finetuned_config,
        top_k=args.top_k,
    )

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
