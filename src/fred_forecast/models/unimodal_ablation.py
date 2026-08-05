"""Unimodal-ablation: unimodal variant of the custom model, without metadata.

Patch encoder -> symbolic coefficients -> SymbolicHead. Although the DataLoader loads the
metadata embeddings (to keep the same batch signature as NeuroSym-CBF), the
`forward` does not touch them: this is deliberate, so that it is exactly NeuroSym-CBF without the
multimodal branch and therefore isolates the contribution of metadata in the ablation.
"""
from __future__ import annotations

import torch.nn as nn

from fred_forecast.models.patch_encoder import PatchEncoder
from fred_forecast.models.symbolic_head import SymbolicHead


class UnimodalAblationModel(nn.Module):
    def __init__(
        self,
        seq_len,
        horizon_len,
        patch_len,
        patch_stride,
        d_model,
        d_ff_ts,
        dropout,
        seasonality,
        num_blocks,
    ):
        super().__init__()
        self.encoder = PatchEncoder(seq_len, patch_len, patch_stride, d_model, dropout, num_blocks)
        self.symbolic_head = SymbolicHead(horizon_len, seasonality)
        self.coeff_layer = nn.Linear(d_model, self.symbolic_head.num_coeffs)

    def encode(self, context):
        return self.encoder(context)

    def get_coeffs(self, hidden):
        return self.coeff_layer(hidden)

    def forward(self, context):
        hidden = self.encode(context)
        coeffs = self.get_coeffs(hidden)
        return self.symbolic_head(coeffs), coeffs
