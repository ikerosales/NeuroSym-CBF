#!/usr/bin/env python
"""Phase 4 of the pipeline: generate T5 embeddings from each series' metadata (title+tags+units).

Requires running Phases 1-3 first (JSON database) and the tag download
(`scripts/02_download_tags.py`). Writes `metadata_embeddings.pkl` -- a derived artifact of the
FRED dataset, never published (because of FRED's terms of use, see README).

Usage:
    python scripts/05_build_metadata_embeddings.py
"""
from __future__ import annotations

import argparse
import logging

from fred_forecast.config import data_dir, load_yaml_config
from fred_forecast.embeddings.text_encoder import build_metadata_embeddings, save_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_yaml_config("configs/data.yaml")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=config["embeddings"]["model_name"])
    parser.add_argument("--batch-size", type=int, default=config["embeddings"]["batch_size"])
    args = parser.parse_args()

    context_months = config["cleaning"]["context_months"]
    database_json = data_dir() / config["paths"]["processed_dir"] / f"fred_database_context_{context_months}.json"
    tags_dir = data_dir() / config["paths"]["tags_dir"]
    output_file = data_dir() / config["paths"]["embeddings_dir"] / "metadata_embeddings.pkl"

    logger.info("Generating metadata embeddings -> %s", output_file)
    df_embeddings = build_metadata_embeddings(
        database_json_path=database_json,
        tags_parquet_path=tags_dir / "fred_tags_output.parquet",
        notes_lookup_path=tags_dir / "tags_notes_lookup.json",
        model_name=args.model_name,
        max_length=config["embeddings"]["max_length"],
        batch_size=args.batch_size,
    )
    save_embeddings(df_embeddings, output_file)


if __name__ == "__main__":
    main()
