"""Paths and constants shared by the unseen-series scripts.

Three sets of FRED series that never entered training:

- `monthly`  monthly series the cleaning phase of the main dataset discarded, each evaluated on its
             own last window;
- `weekly`   natively Weekly series brought to monthly, on the window of the main test;
- `daily`    natively Daily series brought to monthly, on the window of the main test.

Each set keeps its data under `experimental/unseen_series/data/<set>/` (ignored by git, same rules
as the main `data/`).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"

SETS = ("monthly", "weekly", "daily")
SUBPERIOD_SETS = {"weekly": "Weekly", "daily": "Daily"}  # set -> FRED frequency name
CONTEXT_LENS = (8, 30)

# Last month of the main dataset: the horizon of the main test ends here (July 2020 - December 2021).
WINDOW_END = pd.Timestamp("2021-12-01")
MAX_NAN_RATIO = 0.25


def set_dir(set_name: str) -> Path:
    return DATA_DIR / set_name


def database_path(set_name: str, context_len: int) -> Path:
    """Database evaluated for a set and context length.

    The monthly set has a single database (built for the longest context, each series with its
    whole history); the sub-monthly sets have one per context length, because the fixed window they
    must cover depends on it.
    """
    processed = set_dir(set_name) / "processed"
    if set_name == "monthly":
        return processed / "fred_monthly_discarded.json"
    return processed / f"fred_{set_name}_ctx{context_len}.json"


def embeddings_path(set_name: str) -> Path:
    return set_dir(set_name) / "embeddings" / f"metadata_embeddings_{set_name}.pkl"


def tags_dir(set_name: str) -> Path:
    return set_dir(set_name) / "raw" / "tags"
