#!/usr/bin/env python
"""Phase 1 of the pipeline (metadata): download the tags for each series_id from FRED.

Reads the canonical series list from `series_ids/series_ids.json` (the 194,442 series that
make up the final v4 database) and downloads their tags asynchronously, with checkpointing
so the process can be resumed if interrupted.

Usage:
    python scripts/02_download_tags.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from fred_forecast.config import data_dir, get_env, load_yaml_config
from fred_forecast.data.download_tags import download_all_tags

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()  # only for -h/--help
    config = load_yaml_config("configs/data.yaml")
    api_key = get_env("FRED_API_KEY")

    series_ids_file = Path(config["paths"]["series_ids_file"])
    series_ids = json.loads(series_ids_file.read_text())
    logger.info("Loaded %d series from %s", len(series_ids), series_ids_file)

    tags_dir = data_dir() / config["paths"]["tags_dir"]
    asyncio.run(
        download_all_tags(
            series_ids=series_ids,
            api_key=api_key,
            output_file=tags_dir / "fred_tags_output.parquet",
            checkpoint_file=tags_dir / "fred_tags_checkpoint.json",
            notes_file=tags_dir / "tags_notes_lookup.json",
            max_requests_per_minute=config["tags"]["max_requests_per_minute"],
            concurrent_requests=config["tags"]["concurrent_requests"],
            max_retries=config["tags"]["max_retries"],
            batch_size=config["tags"]["batch_size"],
        )
    )


if __name__ == "__main__":
    main()
