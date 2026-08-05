"""Phase 1 of the pipeline: bulk download of FRED observations by release.

For each FRED release, it downloads its observations filtered by frequency and writes them
incrementally into one parquet per release. It is resumable: if `<output_dir>/release_<id>.parquet`
already exists, or if a previous run recorded that release as empty for the requested frequency
(`_empty_releases.json`), it is skipped without requesting it from the API again.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from fred_forecast.data.fred_client import get_all_release_ids, iter_release_observations

logger = logging.getLogger(__name__)


def _load_checked(state_file: Path) -> set[int]:
    if state_file.exists():
        return set(json.loads(state_file.read_text()))
    return set()


def _save_checked(state_file: Path, checked: set[int]) -> None:
    state_file.write_text(json.dumps(sorted(checked)))


def _append_parquet(df: pd.DataFrame, output_file: Path) -> None:
    if not output_file.exists():
        df.to_parquet(output_file, engine="fastparquet")
    else:
        df.to_parquet(output_file, engine="fastparquet", append=True)


def download_release(
    release_id: int,
    freq: str,
    output_file: Path,
    api_key: str,
    include_units: bool = True,
    include_notes: bool = False,
) -> int:
    """Download all pages of `freq` observations for a release, writing to `output_file`."""
    total = 0
    for page_df in iter_release_observations(
        release_id, api_key, freq, include_units=include_units, include_notes=include_notes
    ):
        _append_parquet(page_df, output_file)
        total += len(page_df)
    return total


def download_all_releases(
    freq: str,
    output_dir: Path,
    api_key: str,
    include_units: bool = True,
    include_notes: bool = False,
) -> None:
    """Download the `freq` observations of all FRED releases into `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = output_dir / "_empty_releases.json"
    checked = _load_checked(state_file)

    release_ids = get_all_release_ids(api_key)
    for i, release_id in enumerate(release_ids, start=1):
        output_file = output_dir / f"release_{release_id}.parquet"

        if output_file.exists():
            logger.info("[%d/%d] release %s already downloaded, skipping", i, len(release_ids), release_id)
            continue
        if release_id in checked:
            logger.info(
                "[%d/%d] release %s had no %s series, skipping", i, len(release_ids), release_id, freq.lower()
            )
            continue

        n_obs = download_release(
            release_id, freq, output_file, api_key, include_units=include_units, include_notes=include_notes
        )
        if n_obs > 0:
            logger.info(
                "[%d/%d] release %s: %d %s observations saved", i, len(release_ids), release_id, n_obs, freq.lower()
            )
        else:
            logger.info("[%d/%d] release %s: no %s series", i, len(release_ids), release_id, freq.lower())
            checked.add(release_id)
            _save_checked(state_file, checked)
