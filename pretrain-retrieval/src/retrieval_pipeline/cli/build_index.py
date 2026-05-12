from __future__ import annotations

import argparse

from retrieval_pipeline.config import load_settings
from retrieval_pipeline.indexing.build_index import build_faiss_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    args = parser.parse_args()

    settings = load_settings(args.config)
    build_faiss_index(settings)
    print(f"FAISS index saved to: {settings.paths.index_dir_path}")


if __name__ == "__main__":
    main()
