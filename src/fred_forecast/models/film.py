"""FiLM modulator: produces (γ, β) from the sparse SAE concepts, used to
multiply/additively modulate the symbolic coefficients (`coeffs_mod = γ⊙coeffs_ns + β`).

Verbatim port, used by NeuroSym-CBF (MAE). The quantile variant
(`QuantileFiLMModulator`, for NeuroSym-CBF with pinball loss) lives in
`experimental/neurosym_cbf_pinball/model.py` -- see `experimental/README.md`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FiLMModulator(nn.Module):
    def __init__(self, dict_size, num_coeffs, d_ff=128):
        super().__init__()
        self.num_coeffs = num_coeffs
        self.net = nn.Sequential(
            nn.Linear(dict_size, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, 2 * num_coeffs),
        )
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            bias_init = torch.zeros(2 * num_coeffs)
            bias_init[:num_coeffs] = 1.0  # γ -> 1
            self.net[-1].bias.copy_(bias_init)

    def forward(self, z):
        """z: (B, dict_size) -> gamma (B, C), beta (B, C)"""
        out = self.net(z)
        gamma = out[:, : self.num_coeffs]
        beta = out[:, self.num_coeffs :]
        return gamma, beta
