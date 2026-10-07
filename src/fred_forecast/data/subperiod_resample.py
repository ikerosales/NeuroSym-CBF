"""Bring FRED series whose native frequency is finer than monthly (Weekly, Daily...) to monthly.

Used by the unseen-series experiment (`experimental/unseen_series/`): series of another native
frequency were never part of the main dataset, which is 100% Monthly.

Alignment rule: for each series and each month in its range, take the last value observed on or
before the first day of that month (a backward as-of join against the `YYYY-MM-01` grid). No value
is ever made up -- no averaging and no interpolation, it is always a real observation, only re-dated
to the month it stands for -- which replicates the date convention of FRED's native Monthly series
(dated on day 1). Observations with a NaN value (gaps already present in the original series) are
dropped before the join, so it always picks the last known valid observation.

The output uses the same schema (`series_id, date, value, title, units`) and file naming
(`release_*.parquet`) as Phase 1 of the main pipeline, so `oos_dataset.build_oos_database` can read
it unchanged.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import tqdm

logger = logging.getLogger(__name__)

_RAW_COLUMNS = ["series_id", "date", "value", "title", "units"]
_DICT_COLUMNS = ["series_id", "title", "units"]


def _read_release_file(file: Path) -> pd.DataFrame:
    """Read a `release_*.parquet` with `series_id`/`title`/`units` as dictionary columns
    (categorical in pandas) instead of plain strings.

    Some FRED releases hold tens of millions of rows under a few thousand `series_id`s (a large
    Daily release: 42.6M rows for 9.7k series). Read as plain strings, every text value is
    replicated once per row and can exhaust memory; with `read_dictionary`, pyarrow stores each
    distinct text once.
    """
    try:
        table = pq.read_table(file, columns=_RAW_COLUMNS, read_dictionary=_DICT_COLUMNS)
    except Exception:  # some old release may come without `units`
        table = pq.read_table(file, columns=_RAW_COLUMNS[:-1], read_dictionary=_DICT_COLUMNS[:-1])
        df = table.to_pandas()
        df["units"] = None
        return df
    return table.to_pandas()


def _monthly_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=start.replace(day=1), end=end.replace(day=1), freq="MS")


def resample_series_to_monthly(dates: pd.Series, values: pd.Series) -> pd.DataFrame:
    """One sub-monthly series (`dates`, `values`) -> one row per month in its range.

    `value` of each month = last non-null value observed on or before the first day of that month.
    Months with no earlier valid observation stay NaN (the quality filter downstream handles them).
    """
    df = pd.DataFrame({"date": pd.to_datetime(dates), "value": values}).dropna(subset=["value"])
    if df.empty:
        return pd.DataFrame({"date": pd.DatetimeIndex([]), "value": pd.Series(dtype=float)})

    df = df.sort_values("date")
    grid = pd.DataFrame({"date": _monthly_grid(df["date"].min(), df["date"].max())})
    return pd.merge_asof(grid, df[["date", "value"]], on="date", direction="backward")


def resample_to_monthly(raw_dir: Path, output_dir: Path) -> dict:
    """Read sub-monthly `release_*.parquet` files from `raw_dir`, write their monthly version to
    `output_dir` (same columns and file names, so the result is a valid `raw_dir` for
    `oos_dataset.build_oos_database`).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(raw_dir.glob("release_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No release_*.parquet files in {raw_dir}")

    n_series = 0
    n_series_empty = 0
    for file in tqdm.tqdm(files, desc="resampling to monthly"):
        df = _read_release_file(file)

        out_chunks: list[pd.DataFrame] = []
        for series_id, group in df.groupby("series_id", sort=False, observed=True):
            monthly = resample_series_to_monthly(group["date"], group["value"])
            n_series += 1
            if monthly.empty:
                n_series_empty += 1
                continue
            monthly["series_id"] = str(series_id)
            monthly["title"] = group["title"].iloc[0]
            monthly["units"] = group["units"].iloc[0]
            out_chunks.append(monthly[_RAW_COLUMNS])
        del df

        if out_chunks:
            pd.concat(out_chunks, ignore_index=True).to_parquet(output_dir / file.name, engine="fastparquet")

    summary = {"n_releases": len(files), "n_series": n_series, "n_series_all_nan": n_series_empty}
    logger.info("Resampled to monthly: %s -> %s", summary, output_dir)
    return summary
