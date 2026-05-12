from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median

from retrieval_pipeline.schemas import Document

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - depends on local environment
    plt = None


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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.casefold())


def _word_count(text: str) -> int:
    return len(_tokenize(text))


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0

    sorted_values = sorted(values)
    index = round((len(sorted_values) - 1) * percentile)
    return sorted_values[index]


def _classify_query_type(query_text: str) -> str:
    normalized = re.sub(r"\s+", " ", query_text.casefold().strip())
    if not normalized:
        return "empty"

    if normalized.startswith(TOPIC_PREFIXES):
        return "topic_request"

    first_word = normalized.split(maxsplit=1)[0]
    if first_word in WH_WORDS:
        return "wh_question"
    if first_word in YES_NO_STARTERS:
        return "yes_no_question"
    return "other"


def _build_length_summary(lengths: list[int]) -> dict[str, float]:
    return {
        "count": len(lengths),
        "min": min(lengths, default=0),
        "max": max(lengths, default=0),
        "mean": mean(lengths) if lengths else 0.0,
        "median": median(lengths) if lengths else 0.0,
        "p95": _percentile(lengths, 0.95),
    }


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required to generate EDA charts. "
            "Please install it before running this module."
        )


def _style_axes(axis, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle="--", alpha=0.3)


def _save_histogram(
    values: list[int],
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str = "Count",
    bins: int = 30,
) -> None:
    _require_matplotlib()

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.hist(values, bins=bins, color="#2F6690", edgecolor="white", alpha=0.9)
    _style_axes(axis, title=title, xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_bar_chart(
    labels: list[str],
    values: list[int | float],
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str = "#3A7D44",
    rotation: int = 0,
) -> None:
    _require_matplotlib()

    fig, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(labels, values, color=color, edgecolor="white")
    axis.tick_params(axis="x", rotation=rotation)
    _style_axes(axis, title=title, xlabel=xlabel, ylabel=ylabel)

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.0f}" if isinstance(value, float) else str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_horizontal_bar_chart(
    labels: list[str],
    values: list[int],
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str = "#D17A22",
) -> None:
    _require_matplotlib()

    fig, axis = plt.subplots(figsize=(11, 7))
    bars = axis.barh(labels, values, color=color, edgecolor="white")
    _style_axes(axis, title=title, xlabel=xlabel, ylabel=ylabel)

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {value}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_eda_charts(
    docs_dir: Path,
    queries_file: Path,
    qrels_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    documents = load_documents(docs_dir)
    queries = load_queries(queries_file)
    qrels = load_qrels(qrels_dir) if qrels_dir is not None else {}

    output_dir = output_dir or Path("analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_word_lengths = [_word_count(document.text) for document in documents]
    doc_char_lengths = [len(document.text) for document in documents]
    query_word_lengths = [_word_count(text) for text in queries.values()]
    query_char_lengths = [len(text) for text in queries.values()]

    query_type_counts = Counter(_classify_query_type(text) for text in queries.values())
    relevance_grade_distribution: Counter[int] = Counter()
    relevant_docs_per_query: list[int] = []
    query_terms = Counter()

    for text in queries.values():
        query_terms.update(_tokenize(text))

    for graded_docs in qrels.values():
        relevance_grade_distribution.update(graded_docs.values())
        relevant_docs_per_query.append(sum(1 for grade in graded_docs.values() if grade > 0))

    top_query_terms = query_terms.most_common(15)

    _save_histogram(
        doc_word_lengths,
        output_dir / "document_word_length_distribution.png",
        title="Document Word Length Distribution",
        xlabel="Words per document",
    )
    _save_histogram(
        doc_char_lengths,
        output_dir / "document_char_length_distribution.png",
        title="Document Character Length Distribution",
        xlabel="Characters per document",
    )
    _save_histogram(
        query_word_lengths,
        output_dir / "query_word_length_distribution.png",
        title="Query Word Length Distribution",
        xlabel="Words per query",
    )
    _save_histogram(
        query_char_lengths,
        output_dir / "query_char_length_distribution.png",
        title="Query Character Length Distribution",
        xlabel="Characters per query",
    )
    _save_bar_chart(
        list(query_type_counts.keys()),
        list(query_type_counts.values()),
        output_dir / "query_type_distribution.png",
        title="Query Type Distribution",
        xlabel="Query type",
        ylabel="Count",
        color="#7C4D8B",
        rotation=15,
    )

    if relevant_docs_per_query:
        _save_histogram(
            relevant_docs_per_query,
            output_dir / "relevant_docs_per_query_distribution.png",
            title="Relevant Documents Per Query",
            xlabel="Relevant documents",
        )

    if relevance_grade_distribution:
        _save_bar_chart(
            [str(label) for label in relevance_grade_distribution.keys()],
            list(relevance_grade_distribution.values()),
            output_dir / "relevance_grade_distribution.png",
            title="Relevance Grade Distribution",
            xlabel="Relevance grade",
            ylabel="Count",
            color="#C14953",
        )

    if top_query_terms:
        _save_horizontal_bar_chart(
            [token for token, _ in reversed(top_query_terms)],
            [count for _, count in reversed(top_query_terms)],
            output_dir / "top_query_terms.png",
            title="Top Query Terms",
            xlabel="Frequency",
            ylabel="Token",
        )

    summary = {
        "collection": {
            "document_count": len(documents),
            "query_count": len(queries),
            "qrel_query_count": len(qrels),
        },
        "document_word_lengths": _build_length_summary(doc_word_lengths),
        "document_char_lengths": _build_length_summary(doc_char_lengths),
        "query_word_lengths": _build_length_summary(query_word_lengths),
        "query_char_lengths": _build_length_summary(query_char_lengths),
        "query_type_distribution": dict(query_type_counts),
        "relevance_grade_distribution": dict(relevance_grade_distribution),
        "relevant_docs_per_query": _build_length_summary(relevant_docs_per_query),
        "top_query_terms": [{"term": term, "count": count} for term, count in top_query_terms],
        "output_dir": str(output_dir.resolve()),
    }

    summary_path = output_dir / "dataset_eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate EDA charts for the Cranfield dataset."
    )
    parser.add_argument("--docs-dir", required=True, type=Path, help="Path to DOCS directory.")
    parser.add_argument(
        "--queries-file",
        required=True,
        type=Path,
        help="Path to Cranfield query file.",
    )
    parser.add_argument(
        "--qrels-dir",
        type=Path,
        default=None,
        help="Optional path to REL directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis"),
        help="Directory where EDA charts will be saved.",
    )
    args = parser.parse_args()

    summary = generate_eda_charts(
        docs_dir=args.docs_dir,
        queries_file=args.queries_file,
        qrels_dir=args.qrels_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
