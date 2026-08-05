"""Point metrics shared by the three phases of the custom model.

`masked_*` functions defined identically in the three source variants of the model
(Unimodal-ablation, NeuroSym-CBF, NeuroSym-CBF pinball), factored here into a single module.
"""
from __future__ import annotations

import torch


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = torch.abs(pred - target)
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def masked_mae_per_sample(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = torch.abs(pred - target)
    return (loss * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-8)


def masked_smape_per_sample(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = (torch.abs(target) + torch.abs(pred)) / 2.0 + 1e-8
    smape = torch.abs(target - pred) / denom
    return (smape * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-8)


def masked_mase_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    context: torch.Tensor,
    mask: torch.Tensor,
    seasonality: int = 12,
    min_naive: float = 1e-3,
    padding_value: float = -1,
) -> torch.Tensor:
    """Per-sample MASE against a seasonal naive baseline computed on the context itself
    (excluding padding). `padding_value` was an implicit global variable in the original
    code (always -1) -- here it is an explicit parameter with the same default value,
    with no behavior change.
    """
    mae_fc = (torch.abs(pred - target) * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-8)

    ctx = context.squeeze(-1)
    B = ctx.size(0)
    naive = torch.zeros(B, device=pred.device)
    for b in range(B):
        c = ctx[b]
        valid = c[c > (padding_value - 0.01)]
        if len(valid) > seasonality:
            naive[b] = torch.mean(torch.abs(valid[seasonality:] - valid[:-seasonality]))
        elif len(valid) > 1:
            naive[b] = torch.mean(torch.abs(valid[1:] - valid[:-1]))
        else:
            naive[b] = 1e-8

    valid_mask = naive >= min_naive
    naive = naive.clamp(min=1e-8)
    mase = mae_fc / naive
    return mase[valid_mask]
