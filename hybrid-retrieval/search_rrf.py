from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from rrf import FusedHit, reciprocal_rank_fusion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRADITIONAL_SOURCE = PROJECT_ROOT / "traditional-retrieval" / "source"
PRETRAIN_SOURCE = PROJECT_ROOT / "pretrain-retrieval" / "src"

sys.path.insert(0, str(TRADITIONAL_SOURCE))
sys.path.insert(0, str(PRETRAIN_SOURCE))


@dataclass(slots=True)
class HybridSearchHit:
    doc_id: str
    cross_score: float
    rrf_score: float
    ranks: dict[str, int]
    source_scores: dict[str, float]
    text: str


@dataclass(slots=True)
class FinetunedComponents:
    biencoder: object
    cross_encoder: object


def build_bm25_retriever(documents_csv: Path):
    import pandas as pd
    from utils.BM25_retriever import BM25Retriever
    from utils.inverted_index import InvertedIndex
    from utils.textPreprocessing import TextPreprocessing

    documents = pd.read_csv(documents_csv)
    required_columns = {"id", "content"}
    missing_columns = required_columns - set(documents.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{documents_csv} is missing required column(s): {missing}")

    text_preprocessor = TextPreprocessing()
    processed_docs = text_preprocessor.process_dataframe(documents)
    vocab = text_preprocessor.get_vocab()

    inverted_index = InvertedIndex(processed_docs_df=processed_docs, vocab=vocab)
    inverted_index._build()

    docs_by_id = {
        str(row.id): str(row.content)
        for row in documents[["id", "content"]].itertuples(index=False)
    }
    return BM25Retriever(processed_docs, inverted_index), docs_by_id


def build_finetuned_components(config_path: Path) -> FinetunedComponents:
    from retrieval_pipeline.config import load_settings
    from retrieval_pipeline.models.reranker_model import RerankerModel
    from retrieval_pipeline.retrieval.pipeline import RetrievalPipeline

    settings = load_settings(config_path.resolve())
    pipeline = RetrievalPipeline.from_settings(settings)

    # Hybrid retrieval needs the cross-encoder after RRF, even when the
    # standalone dense pipeline disables reranking in its config.
    if pipeline.reranker is None:
        pipeline.reranker = RerankerModel(
            model_name_or_path=settings.reranking.model_reference,
            device=settings.reranking.device,
        )

    return FinetunedComponents(
        biencoder=pipeline.retriever,
        cross_encoder=pipeline.reranker,
    )


def hybrid_search(
    query: str,
    *,
    bm25,
    finetuned_biencoder,
    cross_encoder,
    docs_by_id: dict[str, str],
    bm25_k: int,
    finetuned_k: int,
    rrf_top_n: int,
    final_k: int,
    rrf_k: int,
):
    with ThreadPoolExecutor(max_workers=2) as executor:
        bm25_future = executor.submit(bm25.search, query, bm25_k)
        finetuned_future = executor.submit(
            finetuned_biencoder.search,
            query=query,
            top_k=finetuned_k,
        )
        bm25_hits = bm25_future.result()
        finetuned_hits = finetuned_future.result()

    fused_hits = reciprocal_rank_fusion(
        {
            "bm25": bm25_hits,
            "finetuned": finetuned_hits,
        },
        rrf_k=rrf_k,
        top_k=rrf_top_n,
    )
    dense_text_by_id = {hit.doc_id: hit.text for hit in finetuned_hits}
    rerank_candidates = _to_rerank_candidates(
        fused_hits=fused_hits,
        dense_text_by_id=dense_text_by_id,
        docs_by_id=docs_by_id,
    )
    reranked_hits = cross_encoder.rerank(
        query=query,
        hits=rerank_candidates,
        top_k=final_k,
    )

    fused_by_id = {hit.doc_id: hit for hit in fused_hits}
    return [
        _to_hybrid_hit(hit, fused_by_id[hit.doc_id])
        for hit in reranked_hits
    ]


def _to_rerank_candidates(
    *,
    fused_hits: list[FusedHit],
    dense_text_by_id: dict[str, str],
    docs_by_id: dict[str, str],
):
    from retrieval_pipeline.schemas import SearchHit

    candidates = []
    for hit in fused_hits:
        text = dense_text_by_id.get(hit.doc_id) or docs_by_id.get(hit.doc_id, "")
        candidates.append(SearchHit(doc_id=hit.doc_id, score=hit.score, text=text))
    return candidates


def _to_hybrid_hit(hit, fused_hit: FusedHit) -> HybridSearchHit:
    return HybridSearchHit(
        doc_id=hit.doc_id,
        cross_score=hit.score,
        rrf_score=fused_hit.score,
        ranks=fused_hit.ranks,
        source_scores=fused_hit.source_scores,
        text=hit.text,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid BM25 + finetuned dense retrieval using RRF."
    )
    parser.add_argument("--query", required=True, help="Input query.")
    parser.add_argument(
        "--documents-csv",
        type=Path,
        default=PROJECT_ROOT / "traditional-retrieval" / "data" / "documents.csv",
        help="CSV with id and content columns for BM25.",
    )
    parser.add_argument(
        "--finetuned-config",
        type=Path,
        default=PROJECT_ROOT / "pretrain-retrieval" / "configs" / "finetuned.json",
        help="Config file for the finetuned retrieval pipeline.",
    )
    parser.add_argument(
        "--bm25-k",
        type=int,
        default=100,
        help="Number of BM25 hits to pass into RRF.",
    )
    parser.add_argument(
        "--finetuned-k",
        type=int,
        default=100,
        help="Number of finetuned hits to pass into RRF.",
    )
    parser.add_argument(
        "--rrf-top-n",
        type=int,
        default=100,
        help="Number of fused RRF candidates to pass into the cross-encoder.",
    )
    parser.add_argument(
        "--final-k",
        type=int,
        default=10,
        help="Number of final cross-encoder reranked hits to print.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF rank constant. Common values are 60 or 100.",
    )
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Print a short document preview from documents.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        bm25, docs_by_id = build_bm25_retriever(args.documents_csv)
        finetuned = build_finetuned_components(args.finetuned_config)
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"Missing dependency: {error.name}. Install project requirements first."
        ) from error
    except LookupError as error:
        raise SystemExit(
            "Missing NLTK data. Run: python -m nltk.downloader punkt stopwords wordnet"
        ) from error

    hits = hybrid_search(
        args.query,
        bm25=bm25,
        finetuned_biencoder=finetuned.biencoder,
        cross_encoder=finetuned.cross_encoder,
        docs_by_id=docs_by_id,
        bm25_k=args.bm25_k,
        finetuned_k=args.finetuned_k,
        rrf_top_n=args.rrf_top_n,
        final_k=args.final_k,
        rrf_k=args.rrf_k,
    )

    print(
        "rank\tdoc_id\tcross_score\trrf_score\tbm25_rank\tfinetuned_rank\t"
        "bm25_score\tfinetuned_score"
    )
    for rank, hit in enumerate(hits, start=1):
        bm25_rank = hit.ranks.get("bm25", "-")
        finetuned_rank = hit.ranks.get("finetuned", "-")
        bm25_score = _format_optional_score(hit.source_scores.get("bm25"))
        finetuned_score = _format_optional_score(hit.source_scores.get("finetuned"))
        print(
            f"{rank}\t{hit.doc_id}\t{hit.cross_score:.6f}\t{hit.rrf_score:.6f}\t"
            f"{bm25_rank}\t{finetuned_rank}\t{bm25_score}\t{finetuned_score}"
        )

        if args.show_text:
            preview = hit.text.replace("\n", " ")[:180]
            print(f"\t{preview}")


def _format_optional_score(score: float | None) -> str:
    return "-" if score is None else f"{score:.6f}"


if __name__ == "__main__":
    main()
