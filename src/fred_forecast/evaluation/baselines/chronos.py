"""Baselines Chronos-Bolt-Base y Chronos-2 (Amazon) — zero-shot, sin fine-tuning.

Ambos reutilizan `UnivariateTestDataset` (misma normalización/padding que el resto del
pipeline) y las métricas de `forecasting_metrics`. El contexto usado en el TFG fue 8 para
Chronos-Bolt y 30 para Chronos-2; aquí es un parámetro, no un valor fijo, ver
`configs/baselines.yaml`.
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

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_chronos_bolt(pipeline, loader, device, seasonality: int, padding_value: float, max_batches=None):
    """Evalúa Chronos-Bolt-Base. `pipeline` = `BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", ...)`."""
    all_mae, all_smape, all_mase = [], [], []
    all_preds, all_targets, all_contexts = [], [], []

    for batch_idx, (context, horizon, mask, stats) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        ctx_in = context.squeeze(-1).to(device)  # (B, L)
        horizon = horizon.to(device)
        mask_f = mask.to(device, dtype=torch.float32)

        forecast = pipeline.predict(ctx_in, prediction_length=horizon.shape[1])
        pred = forecast.median(dim=1).values.unsqueeze(-1).to(device)  # (B, H, 1)

        all_mae.extend(masked_mae_per_sample(pred, horizon, mask_f).cpu().numpy())
        all_smape.extend(masked_smape_per_sample(pred, horizon, mask_f).cpu().numpy())
        all_mase.extend(
            masked_mase_per_sample(
                pred, horizon, context.to(device), mask_f, seasonality=seasonality, padding_value=padding_value
            )
            .cpu()
            .numpy()
        )
        all_preds.append(pred.cpu())
        all_targets.append(horizon.cpu())
        all_contexts.append(context)

    return _aggregate(all_mae, all_smape, all_mase, all_preds, all_targets, all_contexts)


@torch.no_grad()
def evaluate_chronos2(pipeline, loader, device, seasonality: int, padding_value: float, max_batches=None):
    """Evalúa Chronos-2. `pipeline` = `BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=...)`.

    Chronos-2 necesita el contexto con forma (B, n_variates, L) y devuelve una lista de
    tensores por muestra en vez de un único tensor batched — de ahí el `unsqueeze`/`stack`
    extra frente a Chronos-Bolt. El parche de `torch.Tensor.pin_memory` reproduce un
    workaround necesario de la implementación original (incompatibilidad puntual entre el
    `DataLoader` y el pipeline de Chronos-2, no un bug a "arreglar").
    """
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

            forecast = pipeline.predict(ctx_in, prediction_length=horizon.shape[1])
            forecast_tensor = torch.stack(forecast, dim=0)  # (B, 1, num_samples, H)
            pred = forecast_tensor.median(dim=2).values  # (B, 1, H)
            pred = pred.squeeze(1).unsqueeze(-1).to(device)  # (B, H, 1)

            all_mae.extend(masked_mae_per_sample(pred, horizon, mask_f).cpu().numpy())
            all_smape.extend(masked_smape_per_sample(pred, horizon, mask_f).cpu().numpy())
            all_mase.extend(
                masked_mase_per_sample(
                    pred, horizon, context.to(device), mask_f, seasonality=seasonality, padding_value=padding_value
                )
                .cpu()
                .numpy()
            )
            all_preds.append(pred.cpu())
            all_targets.append(horizon.cpu())
            all_contexts.append(context)
    finally:
        torch.Tensor.pin_memory = original_pin_memory

    return _aggregate(all_mae, all_smape, all_mase, all_preds, all_targets, all_contexts)


def _aggregate(all_mae, all_smape, all_mase, all_preds, all_targets, all_contexts) -> dict:
    mae_arr = np.array(all_mae)
    smape_arr = np.array(all_smape)
    mase_arr = np.array(all_mase)

    return {
        "mae_mean": float(mae_arr.mean()),
        "mae_median": float(np.median(mae_arr)),
        "mae_std": float(mae_arr.std()),
        "mae_p25": float(np.percentile(mae_arr, 25)),
        "mae_p75": float(np.percentile(mae_arr, 75)),
        "mae_p90": float(np.percentile(mae_arr, 90)),
        "smape_mean": float(smape_arr.mean()),
        "smape_median": float(np.median(smape_arr)),
        "mase_mean": float(mase_arr.mean()),
        "mase_median": float(np.median(mase_arr)),
        "mae_per_sample": mae_arr,
        "smape_per_sample": smape_arr,
        "mase_per_sample": mase_arr,
        "preds": torch.cat(all_preds),
        "targets": torch.cat(all_targets),
        "contexts": torch.cat(all_contexts),
    }
