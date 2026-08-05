"""Phase 3 of the pipeline: package the cleaned parquets (Phase 2) into a single JSON database.

One record per series: title, observations, start date, length, etc. This is the structure
consumed by the rest of the pipeline (metadata embeddings, training, evaluation). The
`tags` field is always left as `None` here -- tags are downloaded and used separately (see
`download_tags.py` and, later, `embeddings/text_encoder.py`), and are not merged into this JSON.

Note: the Phase 1/2 parquets are downloaded with `include_units=False, include_notes=False` (see
`fred_client.py`), so in practice `units` and `notes` always end up as `None` -- they remain
conditional fields in case their download is re-enabled in the future.
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

import pandas as pd
import tqdm

logger = logging.getLogger(__name__)


def build_series_record(series_id: str, group: pd.DataFrame) -> dict:
    """Build a series record from its already cleaned observations."""
    return {
        "title": group["title"].iloc[0] if "title" in group.columns else None,
        "series_id": series_id,
        "observations": group["value"].tolist(),
        "start_date": group["date"].min().strftime("%Y-%m-%d"),
        "units": group["units"].iloc[0] if "units" in group.columns else None,
        "length": len(group),
        "tags": None,  # tags are downloaded/used separately, see download_tags.py
        "notes": group["notes"].iloc[0] if "notes" in group.columns else None,
    }


def build_database(processed_dir: Path, context_months: int) -> list[dict]:
    """Iterate over all cleaned parquets in `processed_dir` and build the full database.

    The first element of the returned list is a header object with global metadata
    (`context_length`, `release_date`); the rest are per-series records (see `build_series_record`).
    """
    header = {
        "context_length": context_months,
        "release_date": datetime.datetime.now().strftime("%Y-%m-%d"),
    }
    database: list[dict] = [header]

    for release_parquet in tqdm.tqdm(sorted(processed_dir.glob("*.parquet")), desc="construyendo BD"):
        df = pd.read_parquet(release_parquet, engine="fastparquet")
        for series_id, group in df.groupby("series_id"):
            database.append(build_series_record(series_id, group))

    logger.info("Database built: %d series", len(database) - 1)
    return database


def save_database(database: list[dict], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(database, f, indent=4)
    logger.info("Saved to %s", output_file)
