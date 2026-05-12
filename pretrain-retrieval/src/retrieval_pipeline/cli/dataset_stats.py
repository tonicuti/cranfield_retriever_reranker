from __future__ import annotations

import argparse

from retrieval_pipeline.analysis.dataset_stats import build_dataset_stats_report
from retrieval_pipeline.config import load_settings


def _print_length_stats(title: str, stats) -> None:
    print(title)
    print(f"  count: {stats.count}")
    print(
        "  words: "
        f"min={stats.min_words} (id={stats.min_item_id}), "
        f"max={stats.max_words} (id={stats.max_item_id}), avg={stats.avg_words:.2f}, "
        f"median={stats.median_words:.2f}, p95={stats.p95_words}"
    )
    print(
        "  chars: "
        f"min={stats.min_chars} (id={stats.min_chars_item_id}), "
        f"max={stats.max_chars} (id={stats.max_chars_item_id}), avg={stats.avg_chars:.2f}, "
        f"median={stats.median_chars:.2f}, p95={stats.p95_chars}"
    )


def _print_mapping(title: str, mapping: dict) -> None:
    print(title)
    for key, value in mapping.items():
        print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    args = parser.parse_args()

    settings = load_settings(args.config)
    report = build_dataset_stats_report(
        docs_dir=settings.paths.docs_dir_path,
        queries_file=settings.paths.queries_file_path,
        qrels_dir=settings.paths.qrels_dir_path,
    )

    _print_mapping("Collection statistics", report.collection_stats)

    print("Query type statistics")
    for query_type, count in report.query_type_stats.counts.items():
        percentage = report.query_type_stats.percentages[query_type]
        print(f"  {query_type}: {count} ({percentage:.2f}%)")

    _print_length_stats("Document length statistics", report.document_length_stats)
    _print_length_stats("Query length statistics", report.query_length_stats)
    _print_mapping("Qrels statistics", report.qrels_stats)
    _print_mapping("Relevance grade distribution", report.relevance_grade_distribution)
    _print_mapping("Lexical statistics", report.lexical_stats)

    print("Top query starters")
    for token, count in report.top_query_starters:
        print(f"  {token}: {count}")

    print("Top query terms")
    for token, count in report.top_query_terms:
        print(f"  {token}: {count}")

    print(f"Empty document ids: {', '.join(report.empty_document_ids) or 'none'}")


if __name__ == "__main__":
    main()
