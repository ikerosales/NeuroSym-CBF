#!/usr/bin/env python
"""Local convenience utility: consolidates the full FRED dataset into a single parquet/jsonl.

WARNING: the result of this script should NEVER be published or committed -- it is the complete
FRED observations redistributed in another format, which FRED's terms of use do not allow.
It is only intended as a local-work shortcut (one file instead of 114 separate parquets).
It writes under `<DATA_DIR>/processed/`, which is already fully gitignored.

It is not a numbered pipeline step: it is optional and not used by any other script.

Usage:
    python scripts/export_local_dataset.py
"""
from __future__ import annotations

import argparse
import logging

from fred_forecast.config import data_dir, load_yaml_config
from fred_forecast.data.local_dataset_export import build_consolidated_dataset, save_jsonl_gz, save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()  # only for -h/--help
    logger.warning(
        "This export consolidates the FULL FRED dataset (observations included) into a single "
        "file. DO NOT publish or commit it (FRED's terms of use do not allow it)."
    )

    config = load_yaml_config("configs/data.yaml")
    context_months = config["cleaning"]["context_months"]

    database_json = data_dir() / config["paths"]["processed_dir"] / f"fred_database_context_{context_months}.json"
    tags_dir = data_dir() / config["paths"]["tags_dir"]
    output_dir = data_dir() / config["paths"]["processed_dir"]

    df = build_consolidated_dataset(
        database_json_path=database_json,
        tags_parquet_path=tags_dir / "fred_tags_output.parquet",
        notes_lookup_path=tags_dir / "tags_notes_lookup.json",
    )
    save_parquet(df, output_dir / "fred_dataset_consolidated.parquet")
    save_jsonl_gz(df, output_dir / "fred_dataset_consolidated.jsonl.gz")


if __name__ == "__main__":
    main()
