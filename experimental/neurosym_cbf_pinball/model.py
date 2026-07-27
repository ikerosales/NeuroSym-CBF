"""NeuroSym-CBF (pinball): variante probabilística de NeuroSym-CBF, un forecast por cuantil
en vez de un forecast puntual. Trabajo preliminar/futuro, ver `experimental/README.md`.

Reutiliza los componentes ya validados de `fred_forecast` (patch encoder, TopK-SAE); solo lo
específico de la vectorización por cuantil vive aquí (`QuantileSymbolicHead`,
`QuantileFiLMModulator`, `NeuroSymCBFQuantileModel`).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fred_forecast.models.patch_encoder import PatchEncoder
from fred_forecast.models.topk_sae import TopKSAE


class QuantileSymbolicHead(nn.Module):
    """Igual que `fred_forecast.models.symbolic_head.SymbolicHead`, vectorizada sobre una
    dimensión de cuantiles Q. coeffs: (B, Q, num_coeffs) -> forecast (B, H, Q).
    """

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
        """coeffs: (B, Q, num_coeffs) -> (B, H, Q)"""
        c_trend = coeffs[..., :2]
        c_seasonal = coeffs[..., 2:]
        trend_part = torch.matmul(c_trend, self.trend_basis.T)
        full_s = torch.matmul(c_seasonal, self.seasonal_proj.T)
        idx = torch.arange(self.horizon_len, device=coeffs.device) % self.seasonality
        seas_part = full_s[..., idx]
        out = trend_part + seas_part
        return out.transpose(1, 2)


class QuantileFiLMModulator(nn.Module):
    def __init__(self, dict_size, num_coeffs, num_quantiles, d_ff=128):
        super().__init__()
        self.num_coeffs = num_coeffs
        self.num_quantiles = num_quantiles
        out_dim = 2 * num_quantiles * num_coeffs

        self.net = nn.Sequential(
            nn.Linear(dict_size, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            bias_init = torch.zeros(out_dim)
            bias_init[: num_quantiles * num_coeffs] = 1.0  # γ → 1, β → 0
            self.net[-1].bias.copy_(bias_init)

    def forward(self, z):
        """z: (B, dict_size) -> gamma, beta de forma (B, Q, num_coeffs)"""
        B = z.shape[0]
        out = self.net(z)
        out = out.view(B, 2, self.num_quantiles, self.num_coeffs)
        gamma = out[:, 0]
        beta = out[:, 1]
        return gamma, beta


class NeuroSymCBFQuantileModel(nn.Module):
    """
    Forward pass:
        1. context   -> [TS Encoder]                 -> hidden_ts (B, d_model)
        2. hidden_ts (+) metadata -> [fusion proj]    -> fused     (B, d_sae_in)
        3. fused     -> [TopK-SAE]                    -> z
        4. z         -> [FiLM]                        -> γ, β      (B, Q, C)
        5. coeffs_ns = coeff_layer(hidden_ts) viewed as (B, Q, C)
        6. coeffs_mod = γ ⊙ coeffs_ns + β
        7. coeffs_mod -> [QuantileSymbolicHead]        -> forecast  (B, H, Q)

    Loss (calculada fuera del modelo): pinball loss sobre los Q cuantiles.
    """

    def __init__(
        self,
        seq_len: int,
        horizon_len: int,
        d_meta: int,
        num_quantiles: int,
        patch_len: int,
        patch_stride: int,
        d_model: int,
        d_ff_ts: int,
        dropout: float,
        seasonality: int,
        num_blocks: int,
        dict_mult: int,
        top_k: int,
        d_ff_film: int,
    ):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.encoder = PatchEncoder(seq_len, patch_len, patch_stride, d_model, dropout, num_blocks)

        self.symbolic_head = QuantileSymbolicHead(horizon_len, seasonality)
        num_coeffs = self.symbolic_head.num_coeffs
        self.num_coeffs = num_coeffs

        self.coeff_layer = nn.Linear(d_model, num_quantiles * num_coeffs)

        d_sae_in = dict_mult * d_model
        dict_size = dict_mult * d_model
        self.dict_size = dict_size
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model + d_meta, d_sae_in),
            nn.LayerNorm(d_sae_in),
            nn.GELU(),
        )
        self.sae = TopKSAE(d_input=d_sae_in, dict_size=dict_size, k=top_k)
        self.film = QuantileFiLMModulator(
            dict_size=dict_size, num_coeffs=num_coeffs, num_quantiles=num_quantiles, d_ff=d_ff_film
        )

    def encode_ts(self, context):
        return self.encoder(context)

    def forward(self, context, meta):
        """Devuelve: forecast (B, H, Q), coeffs_ns (B, Q, C), coeffs_mod (B, Q, C),
        z (B, dict_size), gamma (B, Q, C), beta (B, Q, C).
        """
        B = context.shape[0]
        hidden = self.encode_ts(context)
        coeffs_ns = self.coeff_layer(hidden).view(B, self.num_quantiles, self.num_coeffs)

        fused = self.fusion_proj(torch.cat([hidden, meta], dim=-1))
        z, _ = self.sae(fused)

        gamma, beta = self.film(z)
        coeffs_mod = gamma * coeffs_ns + beta

        forecast = self.symbolic_head(coeffs_mod)
        return forecast, coeffs_ns, coeffs_mod, z, gamma, beta
