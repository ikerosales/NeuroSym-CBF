#!/usr/bin/env python
"""Phase 4 of the pipeline: generate T5 embeddings from each series' metadata (title+tags+units).

Requires running Phases 1-3 first (JSON database) and the tag download
(`scripts/02_download_tags.py`). Writes `metadata_embeddings.pkl` -- a derived artifact of the
FRED dataset, never published (because of FRED's terms of use, see README).

`--database`, `--tags-parquet`, `--notes-lookup` and `--output` encode another set of series
instead (the unseen-series experiments under `experimental/`). Several tag files and lookups can
be given: the lookups are merged with priority to the first one, so a tag already present in the
training corpus is expanded exactly as it was then.

Usage:
    python scripts/05_build_metadata_embeddings.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fred_forecast.config import data_dir, load_yaml_config
from fred_forecast.embeddings.text_encoder import build_metadata_embeddings, save_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_yaml_config("configs/data.yaml")

    context_months = config["cleaning"]["context_months"]
    tags_dir = data_dir() / config["paths"]["tags_dir"]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default=config["embeddings"]["model_name"])
    parser.add_argument("--batch-size", type=int, default=config["embeddings"]["batch_size"])
    parser.add_argument(
        "--database", type=Path,
        default=data_dir() / config["paths"]["processed_dir"] / f"fred_database_context_{context_months}.json",
        help="Database to take each series' title and units from",
    )
    parser.add_argument(
        "--tags-parquet", type=Path, nargs="+", default=[tags_dir / "fred_tags_output.parquet"],
        help="One or more tag parquets; they are concatenated and de-duplicated by series_id",
    )
    parser.add_argument(
        "--notes-lookup", type=Path, nargs="+", default=[tags_dir / "tags_notes_lookup.json"],
        help="One or more tag -> note lookups, merged with priority to the first one",
    )
    parser.add_argument(
        "--output", type=Path,
        default=data_dir() / config["paths"]["embeddings_dir"] / "metadata_embeddings.pkl",
    )
    args = parser.parse_args()

    database_json = args.database
    output_file = args.output

    logger.info("Generating metadata embeddings -> %s", output_file)
    df_embeddings = build_metadata_embeddings(
        database_json_path=database_json,
        tags_parquet_path=args.tags_parquet,
        notes_lookup_path=args.notes_lookup,
        model_name=args.model_name,
        max_length=config["embeddings"]["max_length"],
        batch_size=args.batch_size,
    )
    save_embeddings(df_embeddings, output_file)


if __name__ == "__main__":
    main()
