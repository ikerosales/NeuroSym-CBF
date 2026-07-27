"""Modulador FiLM: produce (γ, β) a partir de los conceptos dispersos del SAE, usados para
modular multiplicativa/aditivamente los coeficientes simbólicos (`coeffs_mod = γ⊙coeffs_ns + β`).

Puerto verbatim, usado por NeuroSym-CBF (MAE). La variante por cuantil
(`QuantileFiLMModulator`, para NeuroSym-CBF con pérdida pinball) vive en
`experimental/neurosym_cbf_pinball/model.py` — ver `experimental/README.md`.
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
            bias_init[:num_coeffs] = 1.0  # γ → 1
            self.net[-1].bias.copy_(bias_init)

    def forward(self, z):
        """z: (B, dict_size) -> gamma (B, C), beta (B, C)"""
        out = self.net(z)
        gamma = out[:, : self.num_coeffs]
        beta = out[:, self.num_coeffs :]
        return gamma, beta
