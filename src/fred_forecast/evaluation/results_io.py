"""Save/load evaluation results in a consistent format across the project:
aggregated scalar metrics in JSON, per-sample arrays in `.npz`.

This avoids the pattern from the original code of dumping arrays with hundreds of thousands of
elements as lists inside the JSON (`preds_per_sample`, `contexts_per_sample`, ...):
the JSON stays small, readable, and git-diffable; arrays go into a compressed binary,
much faster to load for the aggregation/comparison scripts (`11_aggregate_results.py`).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_SCALAR_TYPES = (int, float, str, bool, type(None))
_JSON_NATIVE_TYPES = (list, dict)


def save_results(
    results: dict,
    json_path: Path,
    npz_path: Path,
    extra_metadata: dict | None = None,
) -> None:
    """Split `results` into scalar fields (-> `json_path`) and array/tensor fields
    (-> `npz_path`, compressed). `extra_metadata` is always added to the JSON (e.g. which
    model/context/seed generated the result).
    """
    scalars: dict = {}
    arrays: dict = {}

    for key, value in results.items():
        if isinstance(value, _SCALAR_TYPES) or isinstance(value, _JSON_NATIVE_TYPES):
            # Small lists/dicts (e.g. fallback series IDs) go directly into the JSON;
            # for large numeric arrays, use np.ndarray/torch.Tensor (-> .npz).
            scalars[key] = value
        elif isinstance(value, torch.Tensor):
            arrays[key] = value.numpy()
        elif isinstance(value, np.ndarray):
            arrays[key] = value
        else:
            logger.warning("Field %r of type %s ignored (neither scalar nor array)", key, type(value))

    if extra_metadata:
        scalars.update(extra_metadata)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(scalars, f, indent=2)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)

    logger.info("Aggregated metrics -> %s", json_path)
    logger.info("Per-sample arrays -> %s", npz_path)


def load_results(json_path: Path, npz_path: Path) -> dict:
    """Recompose the original results dict from the two files."""
    with open(json_path) as f:
        scalars = json.load(f)
    with np.load(npz_path) as npz:
        arrays = {key: npz[key] for key in npz.files}
    return {**scalars, **arrays}
