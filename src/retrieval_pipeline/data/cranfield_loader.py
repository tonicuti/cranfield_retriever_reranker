from __future__ import annotations

from pathlib import Path

from retrieval_pipeline.schemas import Document


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
