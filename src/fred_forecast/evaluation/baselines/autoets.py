"""AutoETS baseline (`statsforecast`): the classic statistical baseline of the work.

Unlike the custom models (PyTorch), AutoETS works on pandas DataFrames in
"long" format (`unique_id`, `ds`, `y`) and `statsforecast` vectorizes/parallellizes the fit per
series internally -- there is no explicit per-series loop or `DataLoader`. The metrics are
recomputed here in numpy (the ones from `forecasting_metrics`, which are for masked torch
tensors, are not reused) -- it is the same math, just operating on numpy arrays
without padding (each window is guaranteed to have the full length by construction).

There are two variants, controlled by WHEN normalization happens relative to fitting the model:
- `normafter`: AutoETS is fit in the original scale; normalization is applied to the
    predictions afterward.
- `normbefore`: the context is normalized first (per-window min-max) before fitting AutoETS.

The context used in the thesis was 8 (not 30, despite the code comment saying otherwise) --
here it is a parameter, not a fixed value; see `configs/baselines.yaml`.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive

logger = logging.getLogger(__name__)


def build_long_frames(df_test: pd.DataFrame, cntxt_len: int, horizon_len: int):
    """Build the long-format DataFrames (original and normalized) + auxiliary arrays
    (normalized context/target, original-scale target, normalization stats).
    """
    rows_orig = []
    rows_normed = []
    contexts = []
    targets = []
    targets_orig = []
    stats = []
    series_ids = []

    base_date = pd.Timestamp("2000-01-01")  # synthetic dates: AutoETS does not care about the real ones

    for _, row in df_test.iterrows():
        sid = row["series_id"]
        obs = row["observations"]

        context = obs[:cntxt_len].astype(np.float32)
        horizon = obs[cntxt_len : cntxt_len + horizon_len].astype(np.float32)

        c_min = float(context.min())
        c_max = float(context.max())
        denom = c_max - c_min if c_max != c_min else 1e-5

        context_norm = (context - c_min) / denom
        horizon_norm = (horizon - c_min) / denom

        clip_value = 8.0
        horizon_norm = np.clip(horizon_norm, -clip_value + 1.0, clip_value)

        dates = pd.date_range(start=base_date, periods=cntxt_len, freq="MS")

        for d, v_orig, v_norm in zip(dates, context, context_norm):
            rows_orig.append((sid, d, float(v_orig)))
            rows_normed.append((sid, d, float(v_norm)))

        contexts.append(context_norm)
        targets.append(horizon_norm)
        targets_orig.append(horizon)
        stats.append([c_min, denom])
        series_ids.append(sid)

    df_long_orig = pd.DataFrame(rows_orig, columns=["unique_id", "ds", "y"])
    df_long_normed = pd.DataFrame(rows_normed, columns=["unique_id", "ds", "y"])

    return (
        df_long_orig,
        df_long_normed,
        np.stack(contexts),
        np.stack(targets),
        np.stack(targets_orig),
        np.array(stats, dtype=np.float32),
        series_ids,
    )


def run_autoets(df_long: pd.DataFrame, horizon_len: int, season_length: int, n_jobs: int):
    """Fit AutoETS on all series in `df_long` (vectorized by `statsforecast`).

    Returns `(preds_dict, fallback_ids)`: predictions by `series_id`, and the IDs that
    fell back to SeasonalNaive (detected by numeric equality with a pure SeasonalNaive fit
    on the same data, not by model introspection).
    """
    sf = StatsForecast(
        models=[AutoETS(season_length=season_length, model="ZZZ")],
        freq="MS",
        n_jobs=n_jobs,
        fallback_model=SeasonalNaive(season_length=season_length),
    )
    logger.info("Fitting AutoETS on %d series...", df_long["unique_id"].nunique())
    forecasts = sf.forecast(df=df_long, h=horizon_len)

    preds_dict = {
        sid: group["AutoETS"].values.astype(np.float32) for sid, group in forecasts.groupby("unique_id")
    }

    sf_naive = StatsForecast(models=[SeasonalNaive(season_length=season_length)], freq="MS", n_jobs=n_jobs)
    naive_forecasts = sf_naive.forecast(df=df_long, h=horizon_len)
    naive_dict = {
        sid: g["SeasonalNaive"].values.astype(np.float32) for sid, g in naive_forecasts.groupby("unique_id")
    }

    fallback_ids = [
        sid for sid in preds_dict if np.allclose(preds_dict[sid], naive_dict[sid], rtol=1e-5, atol=1e-6)
    ]
    return preds_dict, fallback_ids


def mae_per_sample(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs(pred - target).mean(axis=1)


def smape_per_sample(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    denom = (np.abs(target) + np.abs(pred)) / 2.0 + 1e-8
    return (np.abs(target - pred) / denom).mean(axis=1)


def mase_per_sample(
    pred: np.ndarray, target: np.ndarray, context: np.ndarray, seasonality: int = 12, min_naive: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """Filtered MASE (`naive >= min_naive`) + the array of naive baselines used (to
    reuse in the general naive baseline calculation). `context` here already comes without padding
    (full-length windows by construction), unlike the torch functions.
    """
    mae_fc = np.abs(pred - target).mean(axis=1)

    B = context.shape[0]
    naive = np.zeros(B, dtype=np.float32)
    for b in range(B):
        c = context[b]
        if len(c) > seasonality:
            naive[b] = np.mean(np.abs(c[seasonality:] - c[:-seasonality]))
        elif len(c) > 1:
            naive[b] = np.mean(np.abs(c[1:] - c[:-1]))
        else:
            naive[b] = 1e-8

    valid_mask = naive >= min_naive
    naive = np.clip(naive, a_min=1e-8, a_max=None)
    mase = mae_fc / naive
    return mase[valid_mask], naive


def evaluate_autoets(
    df_test: pd.DataFrame,
    variant: str,
    cntxt_len: int,
    horizon_len: int,
    seasonality: int,
    n_jobs: int,
) -> dict:
    """Evaluate one AutoETS variant ('normafter' or 'normbefore') on `df_test`."""
    if variant not in {"normafter", "normbefore"}:
        raise ValueError(f"variant must be 'normafter' or 'normbefore', not {variant!r}")

    (df_long_orig, df_long_normed, contexts_np, targets_np, _targets_orig_np, stats_np, series_ids) = (
        build_long_frames(df_test, cntxt_len, horizon_len)
    )

    df_long = df_long_orig if variant == "normafter" else df_long_normed
    preds_dict, fallback_ids = run_autoets(df_long, horizon_len, seasonality, n_jobs)
    preds = np.stack([preds_dict[sid] for sid in series_ids])

    clip_value = 8.0
    if variant == "normafter":
        c_min = stats_np[:, 0:1]
        denom = stats_np[:, 1:2]
        preds_norm = (preds - c_min) / denom
    else:
        preds_norm = preds
    preds_norm = np.clip(preds_norm, -clip_value + 1.0, clip_value)

    mae_arr = mae_per_sample(preds_norm, targets_np)
    smape_arr = smape_per_sample(preds_norm, targets_np)
    mase_arr, naive_baselines = mase_per_sample(preds_norm, targets_np, contexts_np, seasonality=seasonality)

    logger.info(
        "SeasonalNaive fallbacks: %d (%.1f%%)", len(fallback_ids), 100 * len(fallback_ids) / len(series_ids)
    )

    return {
        "variant": variant,
        "n_test_samples": int(len(mae_arr)),
        "n_fallbacks": int(len(fallback_ids)),
        "fallback_pct": float(len(fallback_ids) / len(series_ids)),
        "mae_mean": float(mae_arr.mean()),
        "mae_median": float(np.median(mae_arr)),
        "mae_std": float(mae_arr.std()),
        "mae_p25": float(np.percentile(mae_arr, 25)),
        "mae_p75": float(np.percentile(mae_arr, 75)),
        "mae_p90": float(np.percentile(mae_arr, 90)),
        "smape_mean": float(smape_arr.mean()),
        "smape_median": float(np.median(smape_arr)),
        "mase_mean": float(mase_arr.mean()) if len(mase_arr) else float("nan"),
        "mase_median": float(np.median(mase_arr)) if len(mase_arr) else float("nan"),
        "mae_per_step": np.abs(preds_norm - targets_np).mean(axis=0),
        "naive_baseline_mean": float(naive_baselines.mean()),
        "naive_baseline_median": float(np.median(naive_baselines)),
        "naive_pct_below_001": float((naive_baselines < 0.01).mean()),
        "mae_per_sample": mae_arr,
        "smape_per_sample": smape_arr,
        "mase_per_sample": mase_arr,
        "preds": preds_norm,
        "targets": targets_np,
        "contexts": contexts_np,
        "fallback_series_ids": fallback_ids,
    }
