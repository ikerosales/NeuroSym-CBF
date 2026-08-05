"""NeuroSym-CBF (MAE): train and evaluate `NeuroSymCBFModel`.

The loss is pure MAE -- the SAE does not add any reconstruction/sparsity term to the loss
(see `TopKSAE` docstring); the only thing new relative to Unimodal-ablation is the call to
`model.sae.normalize_decoder()` after each `optimizer.step()`.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

from fred_forecast.metrics.forecasting_metrics import (
    masked_mae,
    masked_mae_per_sample,
    masked_mase_per_sample,
    masked_smape_per_sample,
)

logger = logging.getLogger(__name__)


def train_neurosym_cbf(model, loader, optimizer, scheduler, device, num_epochs=3, max_batches=None):
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
            loss = masked_mae(forecast, horizon, mask_f)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.sae.normalize_decoder()

            running_total += loss.item()
            if batch_idx % 500 == 0:
                logger.info(
                    "Epoch [%d/%d] | Batch [%d/%d] | loss: %.4f",
                    epoch + 1, num_epochs, batch_idx + 1, len(loader), loss.item(),
                )

        n = batch_idx + 1
        logger.info("Epoch %d/%d | Train MAE: %.4f", epoch + 1, num_epochs, running_total / n)
        scheduler.step()


@torch.no_grad()
def evaluate_neurosym_cbf(model, loader, device, seasonality=12, padding_value=-1, max_batches=None):
    model.eval()
    all_mae, all_smape, all_mase = [], [], []
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

    for batch_idx, (context, horizon, mask, metadata, stats) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        context = context.to(device)
        horizon = horizon.to(device)
        mask_f = mask.to(device, dtype=torch.float32)
        metadata = metadata.to(device)

        forecast, coeffs_ns, coeffs_mod, z, gamma, beta = model(context, metadata)

        all_mae.extend(masked_mae_per_sample(forecast, horizon, mask_f).cpu().numpy())
        all_smape.extend(masked_smape_per_sample(forecast, horizon, mask_f).cpu().numpy())
        all_mase.extend(
            masked_mase_per_sample(
                forecast, horizon, context, mask_f, seasonality=seasonality, padding_value=padding_value
            )
            .cpu()
            .numpy()
        )
        all_preds.append(forecast.cpu())
        all_targets.append(horizon.cpu())
        all_contexts.append(context.cpu())
        all_coeffs_ns.append(coeffs_ns.cpu())
        all_coeffs_mod.append(coeffs_mod.cpu())
        all_z.append(z.cpu())
        all_gamma.append(gamma.cpu())
        all_beta.append(beta.cpu())

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
        "coeffs_ns": torch.cat(all_coeffs_ns),
        "coeffs_mod": torch.cat(all_coeffs_mod),
        "z": torch.cat(all_z),
        "gamma": torch.cat(all_gamma),
        "beta": torch.cat(all_beta),
    }
