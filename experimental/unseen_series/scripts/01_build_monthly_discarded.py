#!/usr/bin/env python
"""Unseen series, step 1: the monthly series the main dataset discarded.

The cleaning phase of the main pipeline dropped thousands of monthly FRED series for not covering
the common window or for having too many gaps. Their observations are still in the raw parquets of
Phase 1, so the set can be rebuilt without downloading anything: this script takes every candidate
`series_id` that did not make it into the final dataset and puts it through the same quality bar
(`fred_forecast.data.oos_dataset`). Each series keeps its whole clean history and is later evaluated
on its own last window.

It only needs the raw monthly parquets and the list of candidate `series_id`s before filtering.

Usage:
    python experimental/unseen_series/scripts/01_build_monthly_discarded.py \\
        --raw-dir <dir with the raw monthly release_*.parquet> --pool-file <candidate_series_ids.json>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from paths import MAX_NAN_RATIO, database_path

from fred_forecast.config import load_yaml_config
from fred_forecast.data.oos_dataset import build_oos_database, load_discarded_ids, save_database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    data_config = load_yaml_config("configs/data.yaml")
    protocol = load_yaml_config("configs/model_protocol.yaml")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True, help="Raw monthly parquets of Phase 1 (release_*.parquet)")
    parser.add_argument("--pool-file", type=Path, required=True, help="JSON with every candidate series_id before filtering")
    parser.add_argument(
        "--kept-file", type=Path, default=Path(data_config["paths"]["series_ids_file"]),
        help="JSON with the series_id that did enter the final dataset",
    )
    args = parser.parse_args()

    # Minimum length for the longest context evaluated; shorter contexts use the same database.
    min_length = protocol["data"]["context_len"] + protocol["data"]["horizon_len"]

    discarded = load_discarded_ids(args.pool_file, args.kept_file)
    database, summary = build_oos_database(
        raw_dir=args.raw_dir, series_ids=discarded, min_length=min_length, max_nan_ratio=MAX_NAN_RATIO
    )

    output_file = database_path("monthly", protocol["data"]["context_len"])
    save_database(database, output_file)
    ids_file = output_file.with_name(output_file.stem + "_series_ids.json")
    ids_file.write_text(json.dumps([r["series_id"] for r in database if "series_id" in r]))
    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
