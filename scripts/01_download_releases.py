#!/usr/bin/env python
"""Phase 1 of the pipeline: bulk download of FRED observations by release.

Downloads, for each frequency configured in configs/data.yaml (Monthly in the thesis), the
observations from ALL FRED releases and saves them incrementally as parquet under
`<DATA_DIR>/raw/parquet_<freq>/release_<id>.parquet`.

Usage:
    python scripts/01_download_releases.py
    python scripts/01_download_releases.py --freq Monthly
"""
from __future__ import annotations

import argparse
import logging

from fred_forecast.config import data_dir, fred_api_key, load_yaml_config
from fred_forecast.data.download_releases import download_all_releases

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_yaml_config("configs/data.yaml")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freq",
        choices=config["frequencies"],
        help="If omitted, downloads all frequencies listed in configs/data.yaml",
    )
    args = parser.parse_args()

    api_key = fred_api_key()
    freqs = [args.freq] if args.freq else config["frequencies"]
    raw_dir = data_dir() / config["paths"]["raw_dir"]

    for freq in freqs:
        output_dir = raw_dir / f"parquet_{freq.strip().lower()}"
        logger.info("Downloading releases at frequency %s -> %s", freq, output_dir)
        download_all_releases(freq=freq, output_dir=output_dir, api_key=api_key)


if __name__ == "__main__":
    main()
