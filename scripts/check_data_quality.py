#!/usr/bin/env python
"""Data quality report for a directory of downloaded release parquets.

It is not a numbered pipeline step (it is an ad hoc diagnostic utility, runnable at
any time after Phase 1) -- it checks duplicate series_ids across releases, the
distribution of last-update dates, and the % of NaN by series.

Usage:
    python scripts/check_data_quality.py --parquet-dir data/raw/parquet_monthly
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fred_forecast.data.quality_checks import (
    find_duplicate_series_ids,
    last_update_stats,
    nan_percentage_by_series,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    args = parser.parse_args()

    duplicates = find_duplicate_series_ids(args.parquet_dir)
    logger.info("Duplicate series_ids across releases: %d", len(duplicates))
    for sid, count in list(duplicates.items())[:10]:
        logger.info("  %s appears %d times", sid, count)

    last_dates = last_update_stats(args.parquet_dir)
    logger.info("Oldest last-update date: %s", last_dates.iloc[0])
    logger.info("5th percentile of last-update dates: %s", last_dates.iloc[int(0.05 * len(last_dates))])
    logger.info("10th percentile of last-update dates: %s", last_dates.iloc[int(0.10 * len(last_dates))])

    nan_pct = nan_percentage_by_series(args.parquet_dir)
    logger.info(
        "%% NaN by series -- describe():\n%s",
        nan_pct.describe(percentiles=[0.5, 0.75, 0.85, 0.9, 0.95, 0.99]),
    )


if __name__ == "__main__":
    main()
