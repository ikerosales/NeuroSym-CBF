"""Out-of-sample datasets: FRED series the models never saw in training.

The train/test split of the main pipeline (`split_train_test_obs`) is temporal: every series
contributes training windows and its last window to the test, so every test `series_id` was seen
during training. That is the usual protocol for a global forecasting model, but it leaves a
question open that this module lets you answer: how does a model do on series it has never seen,
in any window?

The candidates are series that are not in the main database: monthly series the cleaning phase
dropped, or series of another native frequency brought to monthly (`subperiod_resample.py`). Their
observations are rebuilt from raw `release_*.parquet` files and go through the same quality bar as
the series that did enter (trim to the valid range, NaN-ratio filter, forward fill); otherwise any
degradation measured would mix "new series" with "dirtier series".
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tqdm

logger = logging.getLogger(__name__)


def list_raw_series_ids(raw_dir: Path) -> set[str]:
    """Every `series_id` present in the raw `release_*.parquet` files of `raw_dir`.

    Read through pyarrow with a dictionary column, so a release with millions of rows under a few
    thousand series does not replicate the id text once per row.
    """
    files = sorted(raw_dir.glob("release_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No release_*.parquet files in {raw_dir}")
    series_ids: set[str] = set()
    for file in tqdm.tqdm(files, desc="listing raw series"):
        table = pq.read_table(file, columns=["series_id"], read_dictionary=["series_id"])
        series_ids.update(str(s) for s in pc.unique(table.column("series_id")).to_pylist())
    return series_ids


def load_discarded_ids(kept_path: Path, raw_dir: Path | None = None, pool_path: Path | None = None) -> set[str]:
    """`series_id`s of the candidate pool that did not make it into the final dataset.

    The candidate pool is every series in the raw download (`raw_dir`), which is what the cleaning
    phase started from; `pool_path` gives an explicit list instead.
    """
    if pool_path is not None:
        with open(pool_path) as f:
            pool = {str(s) for s in json.load(f)}
    elif raw_dir is not None:
        pool = list_raw_series_ids(raw_dir)
    else:
        raise ValueError("Either raw_dir or pool_path is needed to know the candidate pool")
    with open(kept_path) as f:
        kept = {str(s) for s in json.load(f)}
    discarded = pool - kept
    logger.info("Pool: %d | used: %d | discarded: %d", len(pool), len(kept), len(discarded))
    return discarded


def _collect_raw(raw_dir: Path, series_ids: set[str]) -> tuple[pd.DataFrame, dict[str, tuple]]:
    """Collect (series_id, date, value) from the raw parquets, plus title/units per series.

    The text is kept apart in a dictionary instead of being carried on every row: repeated over
    millions of rows it blows up memory for nothing.
    """
    chunks: list[pd.DataFrame] = []
    text: dict[str, tuple] = {}

    files = sorted(raw_dir.glob("release_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No release_*.parquet files in {raw_dir}")

    for file in tqdm.tqdm(files, desc="reading raw files"):
        columns = ["series_id", "date", "value", "title", "units"]
        try:
            df = pd.read_parquet(file, columns=columns, engine="fastparquet")
        except Exception:  # some old release may come without `units`
            df = pd.read_parquet(file, columns=columns[:-1], engine="fastparquet")
            df["units"] = None

        df = df[df["series_id"].isin(series_ids)]
        if df.empty:
            continue

        for sid, title, units in df.drop_duplicates("series_id")[["series_id", "title", "units"]].itertuples(
            index=False
        ):
            text.setdefault(str(sid), (title, units))

        chunks.append(df[["series_id", "date", "value"]])

    observations = pd.concat(chunks, ignore_index=True)
    observations["series_id"] = observations["series_id"].astype(str)
    logger.info(
        "Raw observations collected: %d rows, %d series", len(observations), observations["series_id"].nunique()
    )
    return observations, text


def build_oos_database(
    raw_dir: Path,
    series_ids: set[str],
    min_length: int,
    max_nan_ratio: float = 0.25,
    window_end: pd.Timestamp | None = None,
) -> tuple[list[dict], dict]:
    """Rebuild the given series with the cleaning rules of the main dataset.

    Per series: trim to [first_valid, last_valid], drop it if the NaN ratio inside that range
    exceeds `max_nan_ratio`, forward-fill the rest, and drop it if fewer than `min_length`
    observations are left (`min_length` should be `context_len + horizon_len`).

    Two modes, depending on `window_end`:

    - `None` (own window): each series keeps its whole clean history and
      `split_train_test_obs` takes its last window, which falls on different dates for each series.
    - `window_end` (fixed window): only the series covering exactly
      `[window_end - (min_length - 1) months, window_end]` are kept, trimmed to that window, so they
      are all evaluated on the same calendar period.

    Returns `(database, summary)` with the same record format as `build_database.py`.
    """
    observations, text = _collect_raw(raw_dir, series_ids)

    window_start = None
    if window_end is not None:
        window_start = window_end - pd.DateOffset(months=min_length - 1)
        logger.info("Fixed window: %s -> %s (%d months)", window_start.date(), window_end.date(), min_length)

    database: list[dict] = [
        {
            "kind": "oos_discarded",
            "window": "fixed" if window_end is not None else "own_last",
            "window_start": str(window_start.date()) if window_start is not None else None,
            "window_end": str(window_end.date()) if window_end is not None else None,
            "min_length": min_length,
            "max_nan_ratio": max_nan_ratio,
            "release_date": datetime.datetime.now().strftime("%Y-%m-%d"),
        }
    ]
    dropped_nan = dropped_short = dropped_empty = dropped_window = 0

    for series_id, group in tqdm.tqdm(observations.groupby("series_id", sort=False), desc="cleaning series"):
        group = group.sort_values("date")
        valid = group["value"].notna()
        if not valid.any():
            dropped_empty += 1
            continue

        group = group.loc[valid.idxmax() : valid[::-1].idxmax()]
        if group["value"].isna().mean() > max_nan_ratio:
            dropped_nan += 1
            continue

        group = group.assign(value=group["value"].ffill())

        if window_end is not None:
            # The forward fill was applied over the whole valid range, so trimming afterwards does
            # not drag gaps in from outside the window.
            if group["date"].min() > window_start or group["date"].max() < window_end:
                dropped_window += 1
                continue
            group = group[(group["date"] >= window_start) & (group["date"] <= window_end)]

        values = group["value"]
        if len(values) < min_length:
            dropped_short += 1
            continue

        title, units = text.get(series_id, (None, None))
        database.append(
            {
                "title": title,
                "series_id": series_id,
                "observations": values.tolist(),
                "start_date": group["date"].min().strftime("%Y-%m-%d"),
                "end_date": group["date"].max().strftime("%Y-%m-%d"),
                "units": units,
                "length": len(values),
                "tags": None,
                "notes": None,
            }
        )

    summary = {
        "n_candidates": len(series_ids),
        "n_with_raw_data": int(observations["series_id"].nunique()),
        "n_final": len(database) - 1,
        "dropped_all_nan": dropped_empty,
        "dropped_nan_ratio": dropped_nan,
        "dropped_outside_window": dropped_window,
        "dropped_too_short": dropped_short,
    }
    logger.info("Out-of-sample dataset built: %s", summary)
    return database, summary


def save_database(database: list[dict], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(database, f)
    logger.info("Saved to %s (%.1f MB)", output_file, output_file.stat().st_size / 1e6)
