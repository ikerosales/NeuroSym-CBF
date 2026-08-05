"""Sparse Autoencoder with hard Top-K activation.

Shared by NeuroSym-CBF (MAE) and NeuroSym-CBF (pinball), with the same code. The
reconstruction (`x_hat`) is computed but not used in any training loss: sparsity is enforced
structurally only via Top-K, not with an L1/reconstruction penalty.
(The original documentation said this reconstruction was used "for the auxiliary loss",
but that is not what the code does; the real behavior has been ported, without adding any
auxiliary loss that was not there.)
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
