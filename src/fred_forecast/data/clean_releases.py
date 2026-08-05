"""Phase 2 of the pipeline: temporal cleaning and homogenization of the downloaded releases.

From the raw per-release parquets (Phase 1), it determines the longest temporal context window
that leaves the largest number of usable series, discards series with too much NaN or with NaN
right at the window boundary, and forward-fills the remaining gaps. The result is one parquet
per release trimmed to that window, ready to be assembled into the final database (see
`build_database.py`). The thesis works only with monthly series.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import tqdm
from fastparquet import ParquetFile

logger = logging.getLogger(__name__)


def compute_valid_bounds(parquet_dir: Path) -> pd.DataFrame:
    """First and last non-NaN date, per series, aggregating across all releases.

    A `series_id` usually lives in a single release, but it is aggregated with min/max in case
    it appears in more than one.
    """
    series_bounds: dict[str, list[pd.Timestamp]] = {}

    for file in tqdm.tqdm(sorted(parquet_dir.glob("release_*.parquet")), desc="valid bounds"):
        df = pd.read_parquet(file, columns=["series_id", "date", "value"], engine="fastparquet")
        g = df.groupby("series_id")
        first_valid = g.apply(lambda x: x.loc[x["value"].notna(), "date"].min())
        last_valid = g.apply(lambda x: x.loc[x["value"].notna(), "date"].max())
        stats = pd.DataFrame({"first_valid": first_valid, "last_valid": last_valid})

        for sid, row in stats.iterrows():
            if sid not in series_bounds:
                series_bounds[sid] = [row["first_valid"], row["last_valid"]]
            else:
                series_bounds[sid][0] = min(series_bounds[sid][0], row["first_valid"])
                series_bounds[sid][1] = max(series_bounds[sid][1], row["last_valid"])

    bounds_valid = pd.DataFrame(series_bounds).T
    bounds_valid.columns = ["first_valid", "last_valid"]
    return bounds_valid


def find_best_cutoff(
    bounds_valid: pd.DataFrame, context_months: int
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Series]:
    """Cutoff date that maximizes the number of series with >= `context_months` of continuous context.

    Devuelve `(best_cutoff, window_start, counts_por_fecha_candidata)`.
    """
    candidate_dates = pd.date_range(
        bounds_valid["first_valid"].min() + pd.DateOffset(months=context_months),
        bounds_valid["last_valid"].max(),
        freq="MS",
    )

    counts = []
    for d in candidate_dates:
        active = (
            (bounds_valid["first_valid"] <= d - pd.DateOffset(months=context_months))
            & (bounds_valid["last_valid"] >= d)
        ).sum()
        counts.append(active)

    counts_series = pd.Series(counts, index=candidate_dates)
    best_cutoff = counts_series.idxmax()
    window_start = best_cutoff - pd.DateOffset(months=context_months)

    logger.info("Best cutoff: %s (%d active series)", best_cutoff, counts_series.max())
    logger.info("Window: %s -> %s", window_start, best_cutoff)
    return best_cutoff, window_start, counts_series


def select_series_with_context(
    bounds_valid: pd.DataFrame, window_start: pd.Timestamp, best_cutoff: pd.Timestamp
) -> set[str]:
    """Series whose valid range fully covers the [window_start, best_cutoff] window."""
    valid = set(
        bounds_valid[
            (bounds_valid["first_valid"] <= window_start) & (bounds_valid["last_valid"] >= best_cutoff)
        ].index
    )
    logger.info("Series with valid context: %d", len(valid))
    return valid


def filter_by_nan_ratio(
    parquet_dir: Path,
    candidate_series: set[str],
    bounds_valid: pd.DataFrame,
    max_nan_ratio: float = 0.25,
) -> set[str]:
    """Discard series whose NaN ratio (within their own [first_valid, last_valid] range)
    exceeds `max_nan_ratio`.
    """
    nan_ratios = []
    for file in tqdm.tqdm(sorted(parquet_dir.glob("release_*.parquet")), desc="ratio de NaN"):
        df = pd.read_parquet(file, columns=["series_id", "date", "value"], engine="fastparquet")
        df = df[df["series_id"].isin(candidate_series)]
        if df.empty:
            continue

        bounds_subset = bounds_valid.loc[df["series_id"].unique()]
        df = df.merge(bounds_subset, left_on="series_id", right_index=True)
        df = df[(df["date"] >= df["first_valid"]) & (df["date"] <= df["last_valid"])]

        g = df.groupby("series_id")["value"].apply(lambda x: x.isna().mean())
        nan_ratios.append(g)

    nan_ratios_combined = pd.concat(nan_ratios).groupby(level=0).mean()
    removed = set(nan_ratios_combined[nan_ratios_combined > max_nan_ratio].index)
    kept = candidate_series - removed

    logger.info("Series discarded for >%.0f%% NaN: %d", max_nan_ratio * 100, len(removed))
    logger.info("Series after NaN filter: %d", len(kept))
    return kept


def filter_by_edge_nan(parquet_dir: Path, candidate_series: set[str], best_cutoff: pd.Timestamp) -> set[str]:
    """Discard series with NaN exactly at the cutoff date (`best_cutoff`)."""
    invalid: set[str] = set()
    for file in tqdm.tqdm(sorted(parquet_dir.glob("release_*.parquet")), desc="NaN at boundary"):
        df = pd.read_parquet(file, columns=["series_id", "date", "value"], engine="fastparquet")
        df = df[df["series_id"].isin(candidate_series)]
        if df.empty:
            continue

        df_cutoff = df[df["date"] == best_cutoff]
        if df_cutoff.empty:
            continue

        bad = df_cutoff[df_cutoff["value"].isna()]["series_id"].unique()
        invalid.update(bad)

    kept = candidate_series - invalid
    logger.info("Series discarded for NaN at the window boundary: %d", len(invalid))
    logger.info("Final usable series: %d", len(kept))
    return kept


def write_cleaned_releases(
    parquet_dir: Path,
    output_dir: Path,
    valid_series: set[str],
    bounds_valid: pd.DataFrame,
    best_cutoff: pd.Timestamp,
) -> None:
    """Trim each release to [first_valid_of_series, best_cutoff], forward-fill the remaining
    NaNs, and write the result (per row-group, so the whole thing is not loaded into memory) to
    `output_dir`.

    Note: forward-fill is applied within each row-group read from the original parquet, not
    over the fully reassembled series -- faithful to the original code behavior. If a series
    history were split across several non-contiguous row-groups, a gap at the start of a
    row-group would not inherit the last valid value from the previous row-group. In practice
    each release fits in a few API pages (500k-row limit), so this was not observed as a problem,
    but it is documented in case the page size changes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    first_valid_map = bounds_valid["first_valid"].to_dict()

    for file in tqdm.tqdm(sorted(parquet_dir.glob("release_*.parquet")), desc="ffill + write"):
        pf = ParquetFile(file)
        output_file = output_dir / file.name
        first_write = True

        for df in pf.iter_row_groups():
            df = df[df["series_id"].isin(valid_series)]
            if df.empty:
                continue

            df["first_valid"] = df["series_id"].map(first_valid_map)
            df = df[(df["date"] >= df["first_valid"]) & (df["date"] <= best_cutoff)]
            df = df.drop(columns="first_valid")
            df = df.sort_values(["series_id", "date"])
            df["value"] = df.groupby("series_id")["value"].transform(lambda s: s.ffill())

            if first_write:
                df.to_parquet(output_file, engine="fastparquet", index=False)
                first_write = False
            else:
                df.to_parquet(output_file, engine="fastparquet", append=True)


def check_context_coverage(
    output_dir: Path, window_start: pd.Timestamp, best_cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Series from the cleaned dataset whose date range does NOT cover [window_start, best_cutoff]
    (should return an empty DataFrame).
    """
    bounds: dict[str, list[pd.Timestamp]] = {}
    for file in tqdm.tqdm(sorted(output_dir.glob("release_*.parquet")), desc="check coverage"):
        df = pd.read_parquet(file, columns=["series_id", "date"])
        g = df.groupby("series_id")["date"].agg(["min", "max"])
        for sid, row in g.iterrows():
            if sid not in bounds:
                bounds[sid] = [row["min"], row["max"]]
            else:
                bounds[sid][0] = min(bounds[sid][0], row["min"])
                bounds[sid][1] = max(bounds[sid][1], row["max"])

    bounds_df = pd.DataFrame(bounds).T
    bounds_df.columns = ["first_date", "last_date"]
    return bounds_df[(bounds_df["first_date"] > window_start) | (bounds_df["last_date"] < best_cutoff)]


def count_remaining_nans(output_dir: Path) -> tuple[int, int]:
    """(number of NaN values, total number of values) in the cleaned dataset -- the first should be 0."""
    total_nans = 0
    total_values = 0
    for file in tqdm.tqdm(sorted(output_dir.glob("release_*.parquet")), desc="check NaN restantes"):
        df = pd.read_parquet(file, columns=["value"])
        total_nans += int(df["value"].isna().sum())
        total_values += len(df)
    return total_nans, total_values


def clean_releases(
    parquet_dir: Path,
    output_dir: Path,
    context_months: int = 48,
    max_nan_ratio: float = 0.25,
) -> dict:
    """Orchestrate the full Phase 2: bounds -> best cutoff -> NaN filters -> ffill + write.

    Returns a summary with the key figures of the process (useful for logging and tests).
    """
    bounds_valid = compute_valid_bounds(parquet_dir)
    best_cutoff, window_start, _counts = find_best_cutoff(bounds_valid, context_months)

    candidate_series = select_series_with_context(bounds_valid, window_start, best_cutoff)
    candidate_series = filter_by_nan_ratio(parquet_dir, candidate_series, bounds_valid, max_nan_ratio)
    valid_series = filter_by_edge_nan(parquet_dir, candidate_series, best_cutoff)

    write_cleaned_releases(parquet_dir, output_dir, valid_series, bounds_valid, best_cutoff)

    violations = check_context_coverage(output_dir, window_start, best_cutoff)
    total_nans, total_values = count_remaining_nans(output_dir)

    summary = {
        "best_cutoff": best_cutoff,
        "window_start": window_start,
        "n_series_final": len(valid_series),
        "n_context_violations": len(violations),
        "n_remaining_nans": total_nans,
        "n_total_values": total_values,
    }
    logger.info("Phase 2 completed: %s", summary)
    if len(violations) > 0:
        logger.warning("%d series do not cover the expected context window", len(violations))
    if total_nans > 0:
        logger.warning("%d NaN values remain in the cleaned dataset", total_nans)

    return summary
