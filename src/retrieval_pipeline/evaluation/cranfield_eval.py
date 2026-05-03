from __future__ import annotations

from math import log2

from retrieval_pipeline.data.cranfield_loader import load_qrels, load_queries
from retrieval_pipeline.retrieval.pipeline import RetrievalPipeline


def _binary_relevant_docs(qrels: dict[str, dict[str, int]], query_id: str) -> set[str]:
    graded = qrels.get(query_id, {})
    return {doc_id for doc_id, grade in graded.items() if grade > 0}


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def average_precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(relevant)


def ndcg_at_k(retrieved: list[str], graded_relevance: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0

    def dcg(scores: list[int]) -> float:
        total = 0.0
        for rank, score in enumerate(scores[:k], start=1):
            total += (2**score - 1) / log2(rank + 1)
        return total

    actual_scores = [graded_relevance.get(doc_id, 0) for doc_id in retrieved[:k]]
    ideal_scores = sorted(
        (score for score in graded_relevance.values() if score > 0),
        reverse=True,
    )

    ideal_dcg = dcg(ideal_scores)
    if ideal_dcg == 0.0:
        return 0.0

    return dcg(actual_scores) / ideal_dcg


def evaluate_pipeline(pipeline: RetrievalPipeline, top_k: int = 10) -> dict[str, float]:
    queries = load_queries(pipeline.settings.paths.queries_file_path)
    qrels = load_qrels(pipeline.settings.paths.qrels_dir_path)

    precision_total = 0.0
    recall_total = 0.0
    map_total = 0.0
    ndcg_total = 0.0
    query_count = 0

    for query_id, query_text in queries.items():
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

    if query_count == 0:
        return {
            "precision@k": 0.0,
            "recall@k": 0.0,
            "map@k": 0.0,
            "ndcg@k": 0.0,
        }

    return {
        "precision@k": precision_total / query_count,
        "recall@k": recall_total / query_count,
        "map@k": map_total / query_count,
        "ndcg@k": ndcg_total / query_count,
    }
