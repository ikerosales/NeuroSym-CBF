"""Symbolic head: evaluates the forecast as intercept + trend + seasonal basis from
coefficients predicted by the encoder.

    ŷ(t) = c₀ + c₁·t̃ + Σₖ aₖ·φₖ(t mod S)      num_coeffs = 1 + 1 + (S - 1)

Used by Unimodal-ablation and NeuroSym-CBF (MAE).
The quantile-vectorized variant (`QuantileSymbolicHead`, for NeuroSym-CBF with pinball loss) lives
in `experimental/neurosym_cbf_pinball/model.py` -- see `experimental/README.md`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SymbolicHead(nn.Module):
    def __init__(self, horizon_len, seasonality=12):
        super().__init__()
        self.horizon_len = horizon_len
        self.seasonality = seasonality
        self.num_coeffs = 1 + 1 + (seasonality - 1)

        t = torch.arange(horizon_len).float()
        t_std = (t - t.mean()) / (t.std() + 1e-8)
        self.register_buffer("trend_basis", torch.stack([torch.ones_like(t), t_std], dim=1))

        S = seasonality
        eye_block = torch.eye(S - 1)
        last_row = -torch.ones(1, S - 1)
        raw = torch.cat([eye_block, last_row], dim=0)
        norm = raw / (raw.norm(dim=0, keepdim=True) + 1e-8)
        self.register_buffer("seasonal_proj", norm)

    def forward(self, coeffs):
        """coeffs: (B, num_coeffs) -> (B, H, 1)"""
        c_trend = coeffs[:, :2]
        c_seasonal = coeffs[:, 2:]
        trend_part = torch.matmul(c_trend, self.trend_basis.T)
        full_s = torch.matmul(c_seasonal, self.seasonal_proj.T)
        idx = torch.arange(self.horizon_len, device=coeffs.device) % self.seasonality
        seas_part = full_s[:, idx]
        return (trend_part + seas_part).unsqueeze(-1)
