"""Baseline TimesFM-200M (Google) — zero-shot, sin fine-tuning.

TimesFM no acepta las ventanas con padding por la izquierda tal cual — su `.forecast()` toma
una lista de arrays 1-D de longitud variable, así
que el padding (`padding_value`) se retira antes de la llamada (heurística de umbral:
`c > padding_value - 0.01`, válida porque los valores reales normalizados nunca bajan de eso
tras el min-max por ventana).
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
def evaluate_timesfm(
    model, loader, device, horizon_len: int, seasonality: int, padding_value: float, max_batches=None
) -> dict:
    """`model` = `timesfm.TimesFm(...)` ya cargado con el checkpoint de HuggingFace."""
    all_mae, all_smape, all_mase = [], [], []
    all_preds, all_targets, all_contexts = [], [], []

    for batch_idx, (context, horizon, mask, stats) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        horizon = horizon.to(device)
        mask_f = mask.to(device, dtype=torch.float32)
        ctx_np = context.squeeze(-1).numpy()  # (B, L)

        context_list = [c[c > (padding_value - 0.01)].astype(np.float32) for c in ctx_np]

        point_preds, _ = model.forecast(context_list, freq=[0] * len(context_list))
        pred_np = point_preds[:, :horizon_len]  # (B, H)
        pred = torch.tensor(pred_np, dtype=torch.float32).unsqueeze(-1).to(device)  # (B, H, 1)

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
