"""Lightweight client for the FRED (Federal Reserve Economic Data) REST API.

Covers the two calls used by the bulk download pipeline: list all release IDs
(API v1) and download, in pages, the observations of a release filtered by frequency
(API v2).

Includes `units` by default: although the last version of the original download driver
disabled it (`include_units=False`), after inspecting the real v4 database used for
the thesis, the `units` field is populated in practically 100% of sampled series
(the releases already existed on disk from a previous download with `include_units=True` and the
driver simply skipped them) -- and it is also used in the metadata embedding text
("Measured in {units}", see Phase 4). `notes` is excluded:
in the real database it is populated inconsistently depending on when each
release was downloaded, and no later phase uses it -- the relevant note information (from tags,
not series) is obtained separately, see `download_tags.py`.
"""
from __future__ import annotations

import logging
import time
from typing import Iterator

import pandas as pd
import requests

logger = logging.getLogger(__name__)

RELEASES_URL = "https://api.stlouisfed.org/fred/releases"
RELEASE_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/v2/release/observations"

VALID_FREQUENCIES = [
    "Daily",
    "Weekly",
    "Biweekly",
    "Monthly",
    "Quarterly",
    "Semiannual",
    "Annual",
    "5 Year Frequency",
]


def get_all_release_ids(api_key: str, page_size: int = 1000) -> list[int]:
    """Return all release IDs available in FRED (paginated)."""
    params: dict = {"api_key": api_key, "file_type": "json", "limit": page_size, "offset": 0}
    releases: list[dict] = []

    while True:
        response = requests.get(RELEASES_URL, params=params)
        response.raise_for_status()
        batch = response.json().get("releases", [])
        releases.extend(batch)

        if len(batch) < page_size:
            break
        params["offset"] += page_size

    release_ids = [r["id"] for r in releases]
    logger.info("Found %d releases", len(release_ids))
    return release_ids


def iter_release_observations(
    release_id: int,
    api_key: str,
    freq: str,
    page_limit: int = 500_000,
    rate_limit_wait_seconds: int = 20,
    include_units: bool = True,
    include_notes: bool = False,
) -> Iterator[pd.DataFrame]:
    """Iterate over the `release_id` observation pages filtered to `freq`.

    Each row is an observation (series_id, date, value, title, and optionally units/notes).
    It retries indefinitely on HTTP 429 (rate limit) while waiting `rate_limit_wait_seconds`;
    for any other HTTP error it stops downloading that release (it logs it and the caller
    continues with the next release).
    """
    if freq not in VALID_FREQUENCIES:
        raise ValueError(f"Invalid frequency {freq!r}. Use one of {VALID_FREQUENCIES}.")

    headers = {"Authorization": f"Bearer {api_key}"}
    params: dict = {"release_id": release_id, "format": "json", "limit": page_limit}
    has_more = True

    while has_more:
        response = requests.get(RELEASE_OBSERVATIONS_URL, params=params, headers=headers)

        if response.status_code == 429:
            logger.warning("Rate limit on release %s, waiting %ds", release_id, rate_limit_wait_seconds)
            time.sleep(rate_limit_wait_seconds)
            continue
        if response.status_code != 200:
            logger.error("HTTP %d downloading release %s", response.status_code, release_id)
            return

        data = response.json()
        rows = []
        for series in data.get("series", []):
            if series.get("frequency") != freq:
                continue
            for obs in series.get("observations", []):
                val = obs.get("value")
                row = {
                    "series_id": series["series_id"],
                    "date": obs["date"],
                    "value": float(val) if val != "." else None,
                    "title": series.get("title"),
                }
                if include_units:
                    row["units"] = series.get("units")
                if include_notes:
                    row["notes"] = series.get("notes")
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            yield df

        has_more = data.get("has_more", False)
        if has_more:
            params["next_cursor"] = data.get("next_cursor")
