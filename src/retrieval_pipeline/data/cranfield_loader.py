from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from retrieval_pipeline.schemas import Document


class TextPreprocessing:
    def __init__(self) -> None:
        self.temp_dataframe: pd.DataFrame | None = None

        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            self.stop_words = set()
        self.lemmatizer = WordNetLemmatizer()

    def _basic_process(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize(self, text: str) -> list[str]:
        return word_tokenize(text, preserve_line=True)

    def _remove_numeric_tokens(self, tokens: list[str]) -> list[str]:
        return [token for token in tokens if not re.search(r"\d", token)]

    def _remove_stopwords(self, tokens: list[str]) -> list[str]:
        return [token for token in tokens if token not in self.stop_words]

    def _lemmatize(self, tokens: list[str]) -> list[str]:
        try:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        except LookupError:
            return tokens

    def _text_preprocessing(self, text: str) -> list[str]:
        text = self._basic_process(text)
        tokens = self._tokenize(text)
        tokens = self._remove_numeric_tokens(tokens)
        tokens = self._remove_stopwords(tokens)
        tokens = self._lemmatize(tokens)
        return tokens

    def process_text(self, text: str) -> str:
        return " ".join(self._text_preprocessing(text))

    def process_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        temp_df = dataframe.copy()

        temp_df["content"] = temp_df["content"].fillna("").astype(str)
        temp_df["tokens"] = (
            temp_df["content"].apply(self._text_preprocessing).apply(tuple)
        )

        self.temp_dataframe = temp_df
        return temp_df

    def get_vocab(self) -> list[str]:
        if self.temp_dataframe is None:
            return []

        vocab: set[str] = set()
        for tokens in self.temp_dataframe["tokens"]:
            vocab.update(tokens)

        return sorted(vocab)


_PREPROCESSOR: TextPreprocessing | None = None


def get_preprocessor() -> TextPreprocessing:
    global _PREPROCESSOR
    if _PREPROCESSOR is None:
        _PREPROCESSOR = TextPreprocessing()
    return _PREPROCESSOR


def preprocess_text(text: Any) -> str:
    return get_preprocessor().process_text(str(text))


def load_documents(docs_dir: Path) -> list[Document]:
    documents: list[Document] = []

    for path in sorted(docs_dir.glob("*.txt"), key=lambda item: int(item.stem)):
        text = preprocess_text(path.read_text(encoding="utf-8", errors="ignore"))
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
            queries[query_id] = preprocess_text(text)

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
