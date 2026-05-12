from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping


@dataclass(slots=True)
class FusedHit:
    doc_id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)
    source_scores: dict[str, float] = field(default_factory=dict)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Iterable[object]],
    *,
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[FusedHit]:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion.

    Each item can be a doc_id string/int, a (doc_id, score) tuple, or an object
    with doc_id and score attributes, such as retrieval_pipeline.schemas.SearchHit.
    """
    fused_scores: dict[str, float] = {}
    ranks_by_doc: dict[str, dict[str, int]] = {}
    scores_by_doc: dict[str, dict[str, float]] = {}

    for source_name, hits in rankings.items():
        seen_in_source: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            doc_id, source_score = _extract_doc_id_and_score(hit)
            if doc_id in seen_in_source:
                continue

            seen_in_source.add(doc_id)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            ranks_by_doc.setdefault(doc_id, {})[source_name] = rank
            if source_score is not None:
                scores_by_doc.setdefault(doc_id, {})[source_name] = source_score

    fused_hits = [
        FusedHit(
            doc_id=doc_id,
            score=score,
            ranks=ranks_by_doc.get(doc_id, {}),
            source_scores=scores_by_doc.get(doc_id, {}),
        )
        for doc_id, score in fused_scores.items()
    ]
    fused_hits.sort(key=lambda hit: (-hit.score, _best_rank(hit), hit.doc_id))

    if top_k is None:
        return fused_hits
    return fused_hits[:top_k]


def _extract_doc_id_and_score(hit: object) -> tuple[str, float | None]:
    if hasattr(hit, "doc_id"):
        doc_id = getattr(hit, "doc_id")
        score = getattr(hit, "score", None)
        return str(doc_id), _as_float_or_none(score)

    if isinstance(hit, tuple) and hit:
        doc_id = hit[0]
        score = hit[1] if len(hit) > 1 else None
        return str(doc_id), _as_float_or_none(score)

    return str(hit), None


def _as_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _best_rank(hit: FusedHit) -> int:
    return min(hit.ranks.values()) if hit.ranks else 10**9
