#!/usr/bin/env python
"""Unseen series, step 3: bring the Weekly or Daily series to monthly frequency.

For every month, the last value observed on or before its first day -- always a real observation,
no averaging or interpolation (see `fred_forecast.data.subperiod_resample`).

Usage:
    python experimental/unseen_series/scripts/03_resample_to_monthly.py --set weekly
"""
from __future__ import annotations

import argparse
import logging

from paths import SUBPERIOD_SETS, set_dir

from fred_forecast.data.subperiod_resample import resample_to_monthly

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=list(SUBPERIOD_SETS), required=True)
    args = parser.parse_args()

    summary = resample_to_monthly(
        raw_dir=set_dir(args.set) / "raw" / f"parquet_{args.set}",
        output_dir=set_dir(args.set) / "interim" / f"{args.set}_monthly",
    )
    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
