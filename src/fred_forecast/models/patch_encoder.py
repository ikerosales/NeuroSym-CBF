"""Patch encoder shared by the three phases of the custom model.

It splits the context into overlapping patches (`unfold`), projects them to `d_model`, and passes
them through `num_blocks` residual blocks. This logic (unfold + input_proj + res_blocks) was
copied identically in the three source model variants (as `encode`/`encode_ts` methods inside each
class) -- factored here into a single class so the code is not triplicated. No operation or tensor
shape has been changed relative to the original.
"""
from __future__ import annotations

import torch.nn as nn

from fred_forecast.models.residual_block import ResidualBlock


class PatchEncoder(nn.Module):
    def __init__(self, seq_len, patch_len, patch_stride, d_model, dropout, num_blocks):
        super().__init__()
        n_patches = (seq_len - patch_len) // patch_stride + 1
        patch_dim = n_patches * patch_len

        self.input_proj = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(d_model, dropout) for _ in range(num_blocks)])
        self.patch_len = patch_len
        self.patch_stride = patch_stride

    def forward(self, context):
        x = context.squeeze(-1)
        x = x.unfold(1, self.patch_len, self.patch_stride).flatten(1)
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        return x
