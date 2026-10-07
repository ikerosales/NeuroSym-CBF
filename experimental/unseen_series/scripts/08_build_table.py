#!/usr/bin/env python
"""Unseen series, step 8: the comparison table, from the per-series errors of step 7.

Writes `results/unseen_series.csv`: mean scaled MAE of every model on the monthly set, on the weekly
and daily sets, and on weekly and daily pooled into one group (the column reported in the paper,
where the per-series errors of both sets are simply put together).

Usage:
    python experimental/unseen_series/scripts/08_build_table.py
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from paths import CONTEXT_LENS, RESULTS_DIR, SETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS = ["neurosym-cbf", "unimodal-ablation", "tft", "chronos-2", "chronos-bolt", "moirai", "timesfm", "autoets-normafter"]


def _errors(model: str, set_name: str, context_len: int) -> np.ndarray | None:
    path = RESULTS_DIR / "metrics" / f"ctx{context_len}" / "npz" / f"{model}-{set_name}.npz"
    if not path.exists():
        return None
    with np.load(path) as npz:
        return npz["mae_per_sample"]


def main() -> None:
    rows = []
    for model in MODELS:
        for context_len in CONTEXT_LENS:
            errors = {s: _errors(model, s, context_len) for s in SETS}
            if any(e is None for e in errors.values()):
                logger.warning("%s ctx%d: results missing, skipped", model, context_len)
                continue
            pooled = np.concatenate([errors["weekly"], errors["daily"]])
            rows.append(
                {
                    "model": model,
                    "context_len": context_len,
                    **{f"n_{s}": len(errors[s]) for s in SETS},
                    **{f"mae_{s}": float(errors[s].mean()) for s in SETS},
                    "n_weekly_daily": len(pooled),
                    "mae_weekly_daily": float(pooled.mean()),
                }
            )
    table = pd.DataFrame(rows)
    out = RESULTS_DIR / "unseen_series.csv"
    table.to_csv(out, index=False)
    logger.info("Table -> %s\n%s", out, table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
