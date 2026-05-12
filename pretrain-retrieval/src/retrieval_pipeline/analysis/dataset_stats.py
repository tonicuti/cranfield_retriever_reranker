from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from statistics import mean, median

from retrieval_pipeline.data.cranfield_loader import load_documents, load_qrels, load_queries


WH_WORDS = {"what", "how", "why", "which", "when", "where", "who", "whom", "whose"}
YES_NO_STARTERS = {
    "is",
    "are",
    "am",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "will",
    "would",
    "should",
    "has",
    "have",
    "had",
}
TOPIC_PREFIXES = (
    "papers on",
    "paper on",
    "studies on",
    "study of",
    "study on",
    "material properties of",
)


@dataclass(slots=True)
class QueryTypeStats:
    counts: dict[str, int]
    percentages: dict[str, float]


@dataclass(slots=True)
class LengthStats:
    count: int
    min_item_id: str
    min_words: int
    max_item_id: str
    max_words: int
    avg_words: float
    median_words: float
    p95_words: int
    min_chars_item_id: str
    min_chars: int
    max_chars_item_id: str
    max_chars: int
    avg_chars: float
    median_chars: float
    p95_chars: int


@dataclass(slots=True)
class DatasetStatsReport:
    query_type_stats: QueryTypeStats
    document_length_stats: LengthStats
    query_length_stats: LengthStats
    collection_stats: dict[str, int]
    qrels_stats: dict[str, float]
    relevance_grade_distribution: dict[int, int]
    lexical_stats: dict[str, int]
    top_query_starters: list[tuple[str, int]]
    top_query_terms: list[tuple[str, int]]
    empty_document_ids: list[str]


def _normalize_text(text: str) -> str:
    normalized = text.casefold().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0

    sorted_values = sorted(values)
    index = round((len(sorted_values) - 1) * percentile)
    return sorted_values[index]


def classify_query_type(query_text: str) -> str:
    normalized = _normalize_text(query_text)
    if not normalized:
        return "empty"

    if normalized.startswith(TOPIC_PREFIXES):
        return "topic_request"

    first_word = normalized.split(maxsplit=1)[0]
    if first_word in WH_WORDS:
        return "wh_question"
    if first_word in YES_NO_STARTERS:
        return "yes_no_question"
    return "keyword_or_statement"


def _build_length_stats(items: list[tuple[str, str]]) -> LengthStats:
    texts = [text for _, text in items]
    word_lengths = [_word_count(text) for text in texts]
    char_lengths = [len(text) for text in texts]

    min_word_item_id = ""
    max_word_item_id = ""
    min_char_item_id = ""
    max_char_item_id = ""

    if items:
        min_word_item_id = min(items, key=lambda item: (_word_count(item[1]), item[0]))[0]
        max_word_item_id = max(items, key=lambda item: (_word_count(item[1]), item[0]))[0]
        min_char_item_id = min(items, key=lambda item: (len(item[1]), item[0]))[0]
        max_char_item_id = max(items, key=lambda item: (len(item[1]), item[0]))[0]

    return LengthStats(
        count=len(texts),
        min_item_id=min_word_item_id,
        min_words=min(word_lengths, default=0),
        max_item_id=max_word_item_id,
        max_words=max(word_lengths, default=0),
        avg_words=mean(word_lengths) if word_lengths else 0.0,
        median_words=median(word_lengths) if word_lengths else 0.0,
        p95_words=_percentile(word_lengths, 0.95),
        min_chars_item_id=min_char_item_id,
        min_chars=min(char_lengths, default=0),
        max_chars_item_id=max_char_item_id,
        max_chars=max(char_lengths, default=0),
        avg_chars=mean(char_lengths) if char_lengths else 0.0,
        median_chars=median(char_lengths) if char_lengths else 0.0,
        p95_chars=_percentile(char_lengths, 0.95),
    )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.casefold())


def _count_words(text: str) -> int:
    return len(_tokenize(text))


def _first_token(text: str) -> str:
    tokens = _tokenize(text)
    return tokens[0] if tokens else "<empty>"


def build_dataset_stats_report(docs_dir, queries_file, qrels_dir=None) -> DatasetStatsReport:
    all_doc_paths = sorted(docs_dir.glob("*.txt"), key=lambda item: int(item.stem))
    empty_document_ids = [
        path.stem
        for path in all_doc_paths
        if not path.read_text(encoding="utf-8", errors="ignore").strip()
    ]

    documents = load_documents(docs_dir)
    queries = load_queries(queries_file)
    qrels = load_qrels(qrels_dir) if qrels_dir is not None else {}

    query_type_counts: dict[str, int] = {
        "wh_question": 0,
        "yes_no_question": 0,
        "topic_request": 0,
        "keyword_or_statement": 0,
        "empty": 0,
    }

    for query_text in queries.values():
        query_type = classify_query_type(query_text)
        query_type_counts[query_type] = query_type_counts.get(query_type, 0) + 1

    total_queries = len(queries)
    query_type_percentages = {
        key: (value / total_queries * 100.0) if total_queries else 0.0
        for key, value in query_type_counts.items()
    }

    document_items = [(document.doc_id, document.text) for document in documents]
    query_items = list(queries.items())
    document_texts = [text for _, text in document_items]
    query_texts = [text for _, text in query_items]
    query_terms = Counter()
    doc_terms = Counter()
    query_starters = Counter()

    for query_text in query_texts:
        query_terms.update(_tokenize(query_text))
        query_starters[_first_token(query_text)] += 1

    for document_text in document_texts:
        doc_terms.update(_tokenize(document_text))

    relevant_counts = []
    relevance_grade_distribution: Counter[int] = Counter()
    for graded_docs in qrels.values():
        relevant_count = sum(1 for grade in graded_docs.values() if grade > 0)
        relevant_counts.append(relevant_count)
        relevance_grade_distribution.update(graded_docs.values())

    qrels_stats = {
        "query_with_qrels_count": len(qrels),
        "avg_relevant_docs_per_query": mean(relevant_counts) if relevant_counts else 0.0,
        "median_relevant_docs_per_query": median(relevant_counts) if relevant_counts else 0.0,
        "min_relevant_docs_per_query": min(relevant_counts, default=0),
        "max_relevant_docs_per_query": max(relevant_counts, default=0),
        "total_qrel_pairs": sum(len(graded_docs) for graded_docs in qrels.values()),
    }

    collection_stats = {
        "raw_document_file_count": len(all_doc_paths),
        "non_empty_document_count": len(documents),
        "empty_document_count": len(empty_document_ids),
        "query_count": len(queries),
    }

    lexical_stats = {
        "query_vocab_size": len(query_terms),
        "document_vocab_size": len(doc_terms),
        "duplicate_query_count": len(query_texts) - len(set(query_texts)),
        "queries_over_20_words": sum(1 for text in query_texts if _count_words(text) > 20),
        "documents_over_300_words": sum(
            1 for text in document_texts if _count_words(text) > 300
        ),
    }

    return DatasetStatsReport(
        query_type_stats=QueryTypeStats(
            counts=query_type_counts,
            percentages=query_type_percentages,
        ),
        document_length_stats=_build_length_stats(document_items),
        query_length_stats=_build_length_stats(query_items),
        collection_stats=collection_stats,
        qrels_stats=qrels_stats,
        relevance_grade_distribution=dict(sorted(relevance_grade_distribution.items())),
        lexical_stats=lexical_stats,
        top_query_starters=query_starters.most_common(10),
        top_query_terms=query_terms.most_common(10),
        empty_document_ids=empty_document_ids,
    )
