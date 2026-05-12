from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Document:
    doc_id: str
    text: str


@dataclass(slots=True)
class SearchHit:
    doc_id: str
    score: float
    text: str
