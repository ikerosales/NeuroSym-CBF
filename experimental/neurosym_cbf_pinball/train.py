"""NeuroSym-CBF (pinball): entrena y evalúa `NeuroSymCBFQuantileModel`.

La pérdida es pinball puro sobre los cuantiles configurados (igual que NeuroSym-CBF con MAE,
el SAE no añade ningún término de reconstrucción/sparsity a la loss). Trabajo
preliminar/futuro, ver `experimental/README.md`.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

from fred_forecast.metrics.forecasting_metrics import (
    masked_mae_per_sample,
    masked_mase_per_sample,
    masked_smape_per_sample,
)
from neurosym_cbf_pinball.metrics import (
    coverage_per_sample,
    interval_width_per_sample,
    pinball_loss_mean,
    pinball_loss_per_sample,
)

logger = logging.getLogger(__name__)


def train_neurosym_cbf_pinball(
    model, loader, optimizer, scheduler, device, quantiles, num_epochs=3, max_batches=None
):
    """`quantiles`: tensor 1-D con los cuantiles a predecir, p.ej. `torch.tensor([0.5, 0.9])`."""
    quantiles = quantiles.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_total = 0.0
        batch_idx = 0
        for batch_idx, (context, horizon, mask, metadata, stats) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            context = context.to(device)
            horizon = horizon.to(device)
            mask_f = mask.to(device, dtype=torch.float32)
            metadata = metadata.to(device)

            optimizer.zero_grad()

            forecast, _, _, _, _, _ = model(context, metadata)
            loss = pinball_loss_mean(forecast, horizon, mask_f, quantiles)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.sae.normalize_decoder()

            running_total += loss.item()
            if batch_idx % 500 == 0:
                logger.info(
                    "Epoch [%d/%d] | Batch [%d/%d] | pinball: %.4f",
                    epoch + 1, num_epochs, batch_idx + 1, len(loader), loss.item(),
                )

        n = batch_idx + 1
        logger.info("Epoch %d/%d | Train pinball: %.4f", epoch + 1, num_epochs, running_total / n)
        scheduler.step()


@torch.no_grad()
def evaluate_neurosym_cbf_pinball(
    model, loader, device, quantiles, median_idx, upper_idx, seasonality=12, padding_value=-1, max_batches=None
):
    """`quantiles`: tensor 1-D de cuantiles. `median_idx`/`upper_idx`: posiciones dentro de
    `quantiles` del cuantil mediano (0.5) y del cuantil superior usado para cobertura/anchura
    de intervalo (0.9 en la configuración del TFG).
    """
    quantiles = quantiles.to(device)
    num_quantiles = len(quantiles)
    model.eval()

    all_pinball, all_cov90, all_width = [], [], []
    all_mae_med, all_smape_med, all_mase_med = [], [], []
    (all_preds, all_targets, all_contexts, all_coeffs_ns, all_coeffs_mod, all_z, all_gamma, all_beta) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    pinball_per_step_sum = None
    pinball_per_step_count = None
    coverage_per_step_sum = None
    coverage_per_step_count = None
    width_per_step_sum = None
    width_per_step_count = None

    for batch_idx, (context, horizon, mask, metadata, stats) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        context = context.to(device)
        horizon = horizon.to(device)
        mask_f = mask.to(device, dtype=torch.float32)
        metadata = metadata.to(device)

        forecast, coeffs_ns, coeffs_mod, z, gamma, beta = model(context, metadata)

        all_pinball.append(pinball_loss_per_sample(forecast, horizon, mask_f, quantiles).cpu().numpy())

        pred_upper = forecast[..., upper_idx : upper_idx + 1]
        pred_median = forecast[..., median_idx : median_idx + 1]
        all_cov90.append(coverage_per_sample(pred_upper, horizon, mask_f).cpu().numpy())
        all_width.append(interval_width_per_sample(pred_upper, pred_median, mask_f).cpu().numpy())

        all_mae_med.append(masked_mae_per_sample(pred_median, horizon, mask_f).cpu().numpy())
        all_smape_med.append(masked_smape_per_sample(pred_median, horizon, mask_f).cpu().numpy())
        all_mase_med.append(
            masked_mase_per_sample(
                pred_median, horizon, context, mask_f, seasonality=seasonality, padding_value=padding_value
            )
            .cpu()
            .numpy()
        )

        target_exp = horizon.expand(-1, -1, num_quantiles)
        diff = target_exp - forecast
        tau = quantiles.view(1, 1, -1)
        pinball_hq = torch.maximum(tau * diff, (tau - 1.0) * diff)
        mask_hq = mask_f.expand_as(pinball_hq)

        step_sum = (pinball_hq * mask_hq).sum(dim=0).cpu().numpy()
        step_cnt = mask_f.squeeze(-1).sum(dim=0).cpu().numpy()

        cov_per_step_sample = (horizon <= pred_upper).float() * mask_f
        cov_step_sum = cov_per_step_sample.squeeze(-1).sum(dim=0).cpu().numpy()

        width_step_sample = (pred_upper - pred_median) * mask_f
        width_step_sum = width_step_sample.squeeze(-1).sum(dim=0).cpu().numpy()

        if pinball_per_step_sum is None:
            pinball_per_step_sum = step_sum
            pinball_per_step_count = step_cnt
            coverage_per_step_sum = cov_step_sum
            coverage_per_step_count = step_cnt.copy()
            width_per_step_sum = width_step_sum
            width_per_step_count = step_cnt.copy()
        else:
            pinball_per_step_sum += step_sum
            pinball_per_step_count += step_cnt
            coverage_per_step_sum += cov_step_sum
            coverage_per_step_count += step_cnt
            width_per_step_sum += width_step_sum
            width_per_step_count += step_cnt

        all_preds.append(forecast.cpu())
        all_targets.append(horizon.cpu())
        all_contexts.append(context.cpu())
        all_coeffs_ns.append(coeffs_ns.cpu())
        all_coeffs_mod.append(coeffs_mod.cpu())
        all_z.append(z.cpu())
        all_gamma.append(gamma.cpu())
        all_beta.append(beta.cpu())

    pinball_arr = np.concatenate(all_pinball, axis=0)  # (N, Q)
    cov90_arr = np.concatenate(all_cov90)
    width_arr = np.concatenate(all_width)
    mae_med_arr = np.concatenate(all_mae_med)
    smape_med_arr = np.concatenate(all_smape_med)
    mase_med_arr = np.concatenate(all_mase_med)

    pinball_per_step = pinball_per_step_sum / (pinball_per_step_count[:, None] + 1e-8)  # (H, Q)
    coverage_per_step = coverage_per_step_sum / (coverage_per_step_count + 1e-8)
    width_per_step = width_per_step_sum / (width_per_step_count + 1e-8)

    return {
        "pinball_mean": float(pinball_arr.mean()),
        "pinball_q50_mean": float(pinball_arr[:, median_idx].mean()),
        "pinball_q90_mean": float(pinball_arr[:, upper_idx].mean()),
        "pinball_q50_median": float(np.median(pinball_arr[:, median_idx])),
        "pinball_q90_median": float(np.median(pinball_arr[:, upper_idx])),
        "pinball_q50_std": float(pinball_arr[:, median_idx].std()),
        "pinball_q90_std": float(pinball_arr[:, upper_idx].std()),
        "pinball_q50_p25": float(np.percentile(pinball_arr[:, median_idx], 25)),
        "pinball_q50_p75": float(np.percentile(pinball_arr[:, median_idx], 75)),
        "pinball_q50_p90": float(np.percentile(pinball_arr[:, median_idx], 90)),
        "pinball_q90_p25": float(np.percentile(pinball_arr[:, upper_idx], 25)),
        "pinball_q90_p75": float(np.percentile(pinball_arr[:, upper_idx], 75)),
        "pinball_q90_p90": float(np.percentile(pinball_arr[:, upper_idx], 90)),
        "coverage_90_mean": float(cov90_arr.mean()),
        "coverage_90_median": float(np.median(cov90_arr)),
        "coverage_90_std": float(cov90_arr.std()),
        "interval_width_mean": float(width_arr.mean()),
        "interval_width_median": float(np.median(width_arr)),
        "interval_width_std": float(width_arr.std()),
        "pct_negative_width": float((width_arr < 0).mean()),
        "mae_median_head_mean": float(mae_med_arr.mean()),
        "mae_median_head_median": float(np.median(mae_med_arr)),
        "mae_median_head_std": float(mae_med_arr.std()),
        "smape_median_head_mean": float(smape_med_arr.mean()),
        "smape_median_head_median": float(np.median(smape_med_arr)),
        "mase_median_head_mean": float(mase_med_arr.mean()),
        "mase_median_head_median": float(np.median(mase_med_arr)),
        "pinball_per_sample": pinball_arr,
        "coverage_per_sample": cov90_arr,
        "width_per_sample": width_arr,
        "mae_median_per_sample": mae_med_arr,
        "smape_median_per_sample": smape_med_arr,
        "pinball_per_step": pinball_per_step,
        "coverage_per_step": coverage_per_step,
        "width_per_step": width_per_step,
        "preds": torch.cat(all_preds),
        "targets": torch.cat(all_targets),
        "contexts": torch.cat(all_contexts),
        "coeffs_ns": torch.cat(all_coeffs_ns),
        "coeffs_mod": torch.cat(all_coeffs_mod),
        "z": torch.cat(all_z),
        "gamma": torch.cat(all_gamma),
        "beta": torch.cat(all_beta),
    }
