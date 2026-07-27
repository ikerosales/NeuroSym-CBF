"""Agrega los resultados de evaluación de todos los modelos (propios y baselines) en tablas
comparativas. Lee los JSON de métricas agregadas (no los `.npz`) generados por
`scripts/10_evaluate.py` y `experimental/*.py`, y los agrupa en dos familias según qué
métricas contienen: forecast puntual (`mae_mean`, ...) vs. probabilístico/pinball
(`pinball_*_mean`, ...) — son familias de métricas distintas, no tiene sentido una única tabla.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

POINT_METRIC_COLUMNS = [
    "model",
    "context_len",
    "horizon_len",
    "n_test_samples",
    "mae_mean",
    "mae_median",
    "mae_std",
    "smape_mean",
    "smape_median",
    "mase_mean",
    "mase_median",
]

PROBABILISTIC_METRIC_COLUMNS = [
    "model",
    "context_len",
    "horizon_len",
    "quantiles",
    "pinball_mean",
    "pinball_q50_mean",
    "pinball_q90_mean",
    "coverage_90_mean",
    "interval_width_mean",
    "mae_median_head_mean",
    "smape_median_head_mean",
    "mase_median_head_mean",
]


def load_all_results(metrics_dir: Path) -> list[dict]:
    """Carga todos los JSON de métricas agregadas de `metrics_dir` (no los `.npz`)."""
    results = []
    for json_path in sorted(metrics_dir.glob("*.json")):
        with open(json_path) as f:
            results.append(json.load(f))
    return results


def build_comparison_tables(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa los resultados en dos tablas: `(tabla_forecast_puntual, tabla_probabilistica)`.

    Un resultado se clasifica como puntual si tiene `mae_mean`, o probabilístico si tiene
    `pinball_mean` o `pinball_q50_mean` (los modelos propios/baselines de punto no tienen
    ninguno de estos dos últimos campos).
    """
    point_rows = [r for r in results if "mae_mean" in r]
    prob_rows = [r for r in results if "pinball_mean" in r or "pinball_q50_mean" in r]

    point_df = pd.DataFrame(point_rows)
    if not point_df.empty:
        cols = [c for c in POINT_METRIC_COLUMNS if c in point_df.columns]
        point_df = point_df[cols].sort_values("mae_mean").reset_index(drop=True)

    prob_df = pd.DataFrame(prob_rows)
    if not prob_df.empty:
        cols = [c for c in PROBABILISTIC_METRIC_COLUMNS if c in prob_df.columns]
        prob_df = prob_df[cols].reset_index(drop=True)

    return point_df, prob_df
