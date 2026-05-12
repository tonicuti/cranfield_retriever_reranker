from __future__ import annotations

import argparse

from retrieval_pipeline.config import load_settings
from retrieval_pipeline.retrieval.pipeline import RetrievalPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument("--query", required=True, help="Input query.")
    args = parser.parse_args()

    settings = load_settings(args.config)
    pipeline = RetrievalPipeline.from_settings(settings)
    hits = pipeline.search(args.query)

    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}\t{hit.doc_id}\t{hit.score:.4f}")


if __name__ == "__main__":
    main()
