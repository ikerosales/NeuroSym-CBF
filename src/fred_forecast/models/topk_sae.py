"""Sparse Autoencoder con activación Top-K dura.

Compartido por NeuroSym-CBF (MAE) y NeuroSym-CBF (pinball), con el mismo código. La
reconstrucción (`x_hat`) se calcula pero no se usa en ninguna pérdida de entrenamiento: la
sparsity se impone solo estructuralmente vía Top-K, no con una penalización L1/reconstrucción.
(La documentación original decía que esta reconstrucción se usaba "para la pérdida auxiliar",
pero no es lo que hace el código; se ha portado el comportamiento real, sin añadir ninguna
pérdida auxiliar que no estaba.)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, d_input, dict_size, k):
        super().__init__()
        self.k = k
        self.b_pre = nn.Parameter(torch.zeros(d_input))
        self.W_enc = nn.Linear(d_input, dict_size, bias=True)
        self.W_dec = nn.Linear(dict_size, d_input, bias=True)
        nn.init.orthogonal_(self.W_enc.weight)
        nn.init.orthogonal_(self.W_dec.weight)

    def encode(self, x):
        x_c = x - self.b_pre
        z_pre = F.relu(self.W_enc(x_c))
        topk_v, topk_i = torch.topk(z_pre, self.k, dim=-1)
        z = torch.zeros_like(z_pre)
        z.scatter_(-1, topk_i, topk_v)
        return z

    def decode(self, z):
        return self.W_dec(z) + self.b_pre

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat

    @torch.no_grad()
    def normalize_decoder(self):
        norms = self.W_dec.weight.norm(dim=1, keepdim=True).clamp(min=1.0)
        self.W_dec.weight.div_(norms)
