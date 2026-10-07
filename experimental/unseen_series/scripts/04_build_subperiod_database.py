#!/usr/bin/env python
"""Unseen series, step 4: common window and quality filter for the Weekly or Daily set.

Reuses `oos_dataset.build_oos_database` in fixed-window mode: each resampled series is trimmed to
`[window_end - (min_length - 1) months, window_end]`, dropped above the NaN ratio, forward-filled,
and dropped if it does not cover the window. `window_end` is the last month of the main dataset
(December 2021), so these series share the calendar window of the main test.

One database per context length (8 and 30): the window a series must cover depends on it, and so
does which series pass the filter. Any `series_id` that is also in the main dataset is excluded.

Usage:
    python experimental/unseen_series/scripts/04_build_subperiod_database.py --set weekly
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from paths import CONTEXT_LENS, MAX_NAN_RATIO, SUBPERIOD_SETS, WINDOW_END, database_path, set_dir

from fred_forecast.config import load_yaml_config
from fred_forecast.data.oos_dataset import build_oos_database, save_database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=list(SUBPERIOD_SETS), required=True)
    args = parser.parse_args()

    data_config = load_yaml_config("configs/data.yaml")
    horizon_len = load_yaml_config("configs/model_protocol.yaml")["data"]["horizon_len"]

    set_ids = {str(s) for s in json.loads((set_dir(args.set) / "raw" / f"{args.set}_series_ids.json").read_text())}
    main_ids = {str(s) for s in json.loads(Path(data_config["paths"]["series_ids_file"]).read_text())}
    overlap = set_ids & main_ids
    if overlap:
        logger.warning("%d series_id are also in the main dataset and are excluded: %s", len(overlap), sorted(overlap)[:10])
    candidate_ids = set_ids - main_ids
    logger.info("Candidate %s series: %d", args.set, len(candidate_ids))

    for context_len in CONTEXT_LENS:
        min_length = context_len + horizon_len
        database, summary = build_oos_database(
            raw_dir=set_dir(args.set) / "interim" / f"{args.set}_monthly",
            series_ids=candidate_ids,
            min_length=min_length,
            max_nan_ratio=MAX_NAN_RATIO,
            window_end=WINDOW_END,
        )
        output_file = database_path(args.set, context_len)
        save_database(database, output_file)
        ids_file = output_file.with_name(output_file.stem + "_series_ids.json")
        ids_file.write_text(json.dumps([r["series_id"] for r in database if "series_id" in r]))
        logger.info("ctx%d -> %s (%d series): %s", context_len, output_file, summary["n_final"], summary)


if __name__ == "__main__":
    main()
