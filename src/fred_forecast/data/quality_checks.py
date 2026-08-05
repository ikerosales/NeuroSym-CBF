"""Quality checks over the downloaded release parquets (ad hoc use, not part
of the sequential pipeline -- see scripts/check_data_quality.py).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


def find_duplicate_series_ids(parquet_dir: Path) -> dict[str, int]:
    """{series_id: number of releases where it appears} for IDs that appear in more than one release."""
    counts: Counter[str] = Counter()
    for file in parquet_dir.glob("release_*.parquet"):
        ids = pd.read_parquet(file, columns=["series_id"], engine="fastparquet")["series_id"].unique()
        counts.update(ids)
    return {sid: n for sid, n in counts.items() if n > 1}


def last_update_stats(parquet_dir: Path) -> pd.Series:
    """Last observation date for each series, across all series in `parquet_dir`."""
    last_dates = []
    for file in parquet_dir.glob("release_*.parquet"):
        df = pd.read_parquet(file, columns=["series_id", "date"], engine="fastparquet")
        last_dates.extend(df.groupby("series_id")["date"].max().tolist())
    return pd.Series(sorted(last_dates))


def nan_percentage_by_series(parquet_dir: Path) -> pd.Series:
    """% of `value` observations that are NaN by series, indexed by series_id."""
    parts = []
    for file in parquet_dir.glob("release_*.parquet"):
        df = pd.read_parquet(file, columns=["series_id", "value"], engine="fastparquet")
        parts.append(df.groupby("series_id")["value"].apply(lambda s: s.isna().mean() * 100))
    return pd.concat(parts)
