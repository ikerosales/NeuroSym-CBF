#!/usr/bin/env python
"""Unseen series, step 2: download every FRED release at Weekly or Daily frequency.

Calls the reusable `download_all_releases` directly with an explicit output folder inside this
experiment, so neither `configs/data.yaml` (Monthly only) nor `DATA_DIR` is touched. It walks all
FRED releases, as the main pipeline does for Monthly, and is resumable in the same way.

Needs `FRED_API_KEY` in `.env`.

Usage:
    python experimental/unseen_series/scripts/02_download_subperiod.py --set weekly
    python experimental/unseen_series/scripts/02_download_subperiod.py --set daily
"""
from __future__ import annotations

import argparse
import json
import logging

import pyarrow.compute as pc
import pyarrow.parquet as pq
from paths import SUBPERIOD_SETS, set_dir

from fred_forecast.config import fred_api_key
from fred_forecast.data.download_releases import download_all_releases

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=list(SUBPERIOD_SETS), required=True)
    args = parser.parse_args()

    frequency = SUBPERIOD_SETS[args.set]
    output_dir = set_dir(args.set) / "raw" / f"parquet_{args.set}"
    logger.info("Downloading %s releases -> %s", frequency, output_dir)
    download_all_releases(freq=frequency, output_dir=output_dir, api_key=fred_api_key())

    # Read through pyarrow with a dictionary column: some Daily releases hold tens of millions of
    # rows under a few thousand series_id, and reading that column as plain strings replicates the
    # text once per row and can exhaust memory.
    series_ids: set[str] = set()
    files = sorted(output_dir.glob("release_*.parquet"))
    for f in files:
        table = pq.read_table(f, columns=["series_id"], read_dictionary=["series_id"])
        series_ids.update(pc.unique(table.column("series_id")).to_pylist())

    ids_file = set_dir(args.set) / "raw" / f"{args.set}_series_ids.json"
    ids_file.write_text(json.dumps(sorted(series_ids)))
    logger.info("%d releases with %s series, %d distinct series_id -> %s", len(files), frequency, len(series_ids), ids_file)


if __name__ == "__main__":
    main()
