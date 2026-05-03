from __future__ import annotations

import argparse

from retrieval_pipeline.config import load_settings
from retrieval_pipeline.evaluation.cranfield_eval import evaluate_pipeline
from retrieval_pipeline.retrieval.pipeline import RetrievalPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument("--top-k", type=int, default=10, help="Metric cutoff.")
    args = parser.parse_args()

    settings = load_settings(args.config)
    pipeline = RetrievalPipeline.from_settings(settings)
    metrics = evaluate_pipeline(pipeline, top_k=args.top_k)

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
