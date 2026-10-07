#!/usr/bin/env python
"""Phase 1 of the pipeline (metadata): download the tags for each series_id from FRED.

Reads the canonical series list from `series_ids.json` (the 194,442 series that
make up the final v4 database) and downloads their tags asynchronously, with checkpointing
so the process can be resumed if interrupted.

`--series-ids-file` and `--prefix` download the tags of another list of series into their own
files (`<prefix>_output.parquet`, `<prefix>_checkpoint.json`, `<prefix>_notes_lookup.json`) without
touching the main download. The unseen-series experiments under `experimental/` use them.

Usage:
    python scripts/02_download_tags.py
    python scripts/02_download_tags.py --series-ids-file ids.json --prefix weekly
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from fred_forecast.config import data_dir, fred_api_key, load_yaml_config
from fred_forecast.data.download_tags import download_all_tags

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_yaml_config("configs/data.yaml")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--series-ids-file", type=Path, default=Path(config["paths"]["series_ids_file"]),
        help="JSON with the list of series_id whose tags to download",
    )
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="Prefix of the output files, so another download does not overwrite the main one",
    )
    args = parser.parse_args()
    api_key = fred_api_key()

    series_ids = json.loads(args.series_ids_file.read_text())
    logger.info("Loaded %d series from %s", len(series_ids), args.series_ids_file)

    tags_dir = data_dir() / config["paths"]["tags_dir"]
    if args.prefix is None:
        output_name, checkpoint_name, notes_name = (
            "fred_tags_output.parquet", "fred_tags_checkpoint.json", "tags_notes_lookup.json",
        )
    else:
        output_name, checkpoint_name, notes_name = (
            f"{args.prefix}_output.parquet", f"{args.prefix}_checkpoint.json", f"{args.prefix}_notes_lookup.json",
        )
    asyncio.run(
        download_all_tags(
            series_ids=series_ids,
            api_key=api_key,
            output_file=tags_dir / output_name,
            checkpoint_file=tags_dir / checkpoint_name,
            notes_file=tags_dir / notes_name,
            max_requests_per_minute=config["tags"]["max_requests_per_minute"],
            concurrent_requests=config["tags"]["concurrent_requests"],
            max_retries=config["tags"]["max_retries"],
            batch_size=config["tags"]["batch_size"],
        )
    )


if __name__ == "__main__":
    main()
