#!/usr/bin/env python
"""Phase 2 of the pipeline: temporal cleaning and homogenization of the downloaded releases.

Determines the context window (48 months by default, see configs/data.yaml:cleaning) that
maximizes the number of usable series, discards those with too much NaN or NaN right at the
window boundary, and forward-fills the rest. Writes the result under
`<DATA_DIR>/interim/final_dataset_<freq>/`.

Usage:
    python scripts/03_clean_releases.py
    python scripts/03_clean_releases.py --freq Monthly
"""
from __future__ import annotations

import argparse
import logging

from fred_forecast.config import data_dir, load_yaml_config
from fred_forecast.data.clean_releases import clean_releases

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_yaml_config("configs/data.yaml")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freq",
        default="Monthly",
        help="Frequency to clean. The thesis uses 'Monthly' (the default value).",
    )
    args = parser.parse_args()

    raw_dir = data_dir() / config["paths"]["raw_dir"] / f"parquet_{args.freq.strip().lower()}"
    output_dir = data_dir() / config["paths"]["interim_dir"] / f"final_dataset_{args.freq.strip().lower()}"

    logger.info("Cleaning %s -> %s", raw_dir, output_dir)
    clean_releases(
        parquet_dir=raw_dir,
        output_dir=output_dir,
        context_months=config["cleaning"]["context_months"],
        max_nan_ratio=config["cleaning"]["max_nan_ratio"],
    )


if __name__ == "__main__":
    main()
