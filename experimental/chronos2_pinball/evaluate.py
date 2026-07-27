"""Baseline Chronos-2 en modo probabilístico (pinball). Trabajo preliminar/futuro, ver
`experimental/README.md` — pensado específicamente para comparar contra NeuroSym-CBF (pinball).

A diferencia del baseline puntual de Chronos-2 (que solo extrae la mediana de las muestras),
aquí se piden los cuantiles 0.5 y 0.9 vía `torch.quantile` sobre las muestras devueltas por
Chronos-2, y se evalúa con las mismas métricas de pinball/cobertura/anchura de intervalo que
`neurosym_cbf_pinball` (reutilizadas de ahí, no reimplementadas otra vez).

Este módulo solo evalúa y devuelve los arrays (incluido `contexts`); la generación de figuras,
si se hace, iría aparte a partir de ese resultado.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from fred_forecast.metrics.forecasting_metrics import (
    masked_mae_per_sample,
    masked_mase_per_sample,
    masked_smape_per_sample,
)
from neurosym_cbf_pinball.metrics import coverage_per_sample, interval_width_per_sample, pinball_loss_per_sample

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_chronos2_pinball(
    pipeline,
    loader,
    device,
    horizon_len: int,
    quantiles: torch.Tensor,
    seasonality: int,
    padding_value: float,
    max_batches=None,
) -> dict:
    """`pipeline` = `BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=...)`.
    `quantiles`: tensor 1-D, p.ej. `torch.tensor([0.5, 0.9])` (q50 = mediana, q90 = superior).
    """
    quantiles = quantiles.to(device)
    all_pinball, all_cov90, all_width = [], [], []
    all_mae, all_smape, all_mase = [], [], []
    all_preds, all_targets, all_contexts = [], [], []

    original_pin_memory = torch.Tensor.pin_memory
    torch.Tensor.pin_memory = lambda self, *args, **kwargs: self
    try:
        for batch_idx, (context, horizon, mask, stats) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            ctx_in = context.squeeze(-1).to(device)  # (B, L)
            ctx_in = ctx_in.unsqueeze(1)  # (B, 1, L) — Chronos-2 quiere n_variates explícito
            horizon = horizon.to(device)
            mask_f = mask.to(device, dtype=torch.float32)

            forecast = pipeline.predict(ctx_in, prediction_length=horizon_len)
            samples = torch.stack(forecast, dim=0).squeeze(1).to(device)  # (B, num_samples, H)

            q50 = torch.quantile(samples, 0.5, dim=1)  # (B, H)
            q90 = torch.quantile(samples, 0.9, dim=1)  # (B, H)
            pred = torch.stack([q50, q90], dim=-1)  # (B, H, 2)

            pred_median = pred[:, :, 0:1]
            pred_upper = pred[:, :, 1:2]

            all_pinball.extend(pinball_loss_per_sample(pred, horizon, mask_f, quantiles).cpu().numpy())
            all_cov90.extend(coverage_per_sample(pred_upper, horizon, mask_f).cpu().numpy())
            all_width.extend(interval_width_per_sample(pred_upper, pred_median, mask_f).cpu().numpy())
            all_mae.extend(masked_mae_per_sample(pred_median, horizon, mask_f).cpu().numpy())
            all_smape.extend(masked_smape_per_sample(pred_median, horizon, mask_f).cpu().numpy())
            all_mase.extend(
                masked_mase_per_sample(
                    pred_median, horizon, context.to(device), mask_f,
                    seasonality=seasonality, padding_value=padding_value,
                )
                .cpu()
                .numpy()
            )
            all_preds.append(pred.cpu())
            all_targets.append(horizon.cpu())
            all_contexts.append(context.cpu())
    finally:
        torch.Tensor.pin_memory = original_pin_memory

    pinball_arr = np.array(all_pinball)  # (N, 2)
    cov_arr = np.array(all_cov90)
    width_arr = np.array(all_width)
    mae_arr = np.array(all_mae)
    smape_arr = np.array(all_smape)
    mase_arr = np.array(all_mase)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    quantile_crossing_frac = float((preds[:, :, 1] < preds[:, :, 0]).float().mean())

    return {
        "pinball_q50_mean": float(pinball_arr[:, 0].mean()),
        "pinball_q90_mean": float(pinball_arr[:, 1].mean()),
        "coverage_90_mean": float(cov_arr.mean()),
        "interval_width_mean": float(width_arr.mean()),
        "mae_median_head_mean": float(mae_arr.mean()),
        "smape_median_head_mean": float(smape_arr.mean()),
        "mase_median_head_mean": float(mase_arr.mean()),
        "quantile_crossing_frac": quantile_crossing_frac,
        "pinball_per_sample": pinball_arr,
        "coverage_per_sample": cov_arr,
        "width_per_sample": width_arr,
        "preds": preds,
        "targets": targets,
        "contexts": torch.cat(all_contexts),
    }
