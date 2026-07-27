"""Métricas probabilísticas (pinball loss, cobertura, anchura de intervalo) para la variante
por cuantiles. Trabajo preliminar/futuro, ver `experimental/README.md`.
"""
from __future__ import annotations

import torch


def pinball_loss_per_sample(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    """
    pred  : (B, H, Q)
    target: (B, H, 1)
    mask  : (B, H, 1) — float
    quantiles: tensor de forma (Q,)
    Devuelve pinball por muestra y por cuantil: (B, Q)
    """
    target_exp = target.expand(-1, -1, len(quantiles))  # (B, H, Q)
    diff = target_exp - pred  # (B, H, Q)
    tau = quantiles.view(1, 1, -1).to(pred.device)  # (1, 1, Q)
    loss = torch.maximum(tau * diff, (tau - 1.0) * diff)  # (B, H, Q)
    mask_exp = mask.expand_as(loss).float()
    per_sample = (loss * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1) + 1e-8)
    return per_sample  # (B, Q)


def pinball_loss_mean(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    """Pinball loss escalar, promediado sobre batch, horizonte y cuantiles."""
    return pinball_loss_per_sample(pred, target, mask, quantiles).mean()


def coverage_per_sample(pred_upper: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cobertura unilateral: fracción de pasos del horizonte (no enmascarados) donde
    `target <= pred_upper`. pred_upper, target, mask: (B, H, 1). Devuelve (B,).
    """
    below = (target <= pred_upper).float()
    return (below * mask.float()).sum(dim=(1, 2)) / (mask.float().sum(dim=(1, 2)) + 1e-8)


def interval_width_per_sample(
    pred_upper: torch.Tensor, pred_median: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Anchura media (q0.9 - q0.5) sobre el horizonte no enmascarado. (B,)."""
    w = pred_upper - pred_median
    return (w * mask.float()).sum(dim=(1, 2)) / (mask.float().sum(dim=(1, 2)) + 1e-8)
