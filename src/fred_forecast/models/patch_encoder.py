"""Encoder de parcheo compartido por las tres fases del modelo propio.

Divide el contexto en parches solapados (`unfold`), los proyecta a `d_model` y los pasa
por `num_blocks` bloques residuales. Esta lógica (unfold + input_proj + res_blocks) estaba
copiada de forma idéntica en las tres variantes de origen del modelo (como método
`encode`/`encode_ts` dentro de cada clase) — factorizada aquí en una única clase para no
triplicar el código. No se ha cambiado ninguna operación ni forma de tensor respecto al original.
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
