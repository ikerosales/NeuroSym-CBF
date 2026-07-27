"""NeuroSym-CBF: modelo propio multimodal (Concept Bottleneck Model + FiLM).

Fusiona la codificación de la serie con los embeddings de metadatos, extrae conceptos
dispersos vía un Top-K Sparse Autoencoder, y usa esos conceptos para modular (FiLM) los
coeficientes simbólicos. Es el modelo principal del TFG.

La variante con pérdida pinball (`NeuroSymCBFQuantileModel`, forecast por cuantil en vez de
puntual) es trabajo preliminar/futuro y vive aparte, en
`experimental/neurosym_cbf_pinball/model.py` — ver `experimental/README.md`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fred_forecast.models.film import FiLMModulator
from fred_forecast.models.patch_encoder import PatchEncoder
from fred_forecast.models.symbolic_head import SymbolicHead
from fred_forecast.models.topk_sae import TopKSAE


class NeuroSymCBFModel(nn.Module):
    """
    Forward pass:
        1. context   -> [TS Encoder]                -> hidden_ts  (B, d_model)
        2. hidden_ts (+) metadata -> [fusion proj]   -> fused      (B, d_sae_in)
        3. fused     -> [TopK-SAE]                   -> z
        4. z         -> [FiLM MLP]                   -> (γ, β)
        5. coeffs_ns = coeff_layer(hidden_ts)
        6. coeffs_mod = γ ⊙ coeffs_ns + β
        7. coeffs_mod -> [SymbolicHead]               -> forecast   (B, H, 1)

    Loss (calculada fuera del modelo): MAE(forecast, horizon).
    """

    def __init__(
        self,
        seq_len: int,
        horizon_len: int,
        d_meta: int,
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
        self.encoder = PatchEncoder(seq_len, patch_len, patch_stride, d_model, dropout, num_blocks)

        self.symbolic_head = SymbolicHead(horizon_len, seasonality)
        num_coeffs = self.symbolic_head.num_coeffs
        self.coeff_layer = nn.Linear(d_model, num_coeffs)

        d_sae_in = dict_mult * d_model  # reuse dict_mult as SAE input dim
        dict_size = dict_mult * d_model
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model + d_meta, d_sae_in),
            nn.LayerNorm(d_sae_in),
            nn.GELU(),
        )
        self.sae = TopKSAE(d_input=d_sae_in, dict_size=dict_size, k=top_k)
        self.film = FiLMModulator(dict_size=dict_size, num_coeffs=num_coeffs, d_ff=d_ff_film)

    def encode_ts(self, context):
        return self.encoder(context)

    def forward(self, context, meta):
        """Devuelve: forecast (B, H, 1), coeffs_ns (B, C), coeffs_mod (B, C),
        z (B, dict_size), gamma (B, C), beta (B, C).
        """
        hidden = self.encode_ts(context)
        coeffs_ns = self.coeff_layer(hidden)

        fused = self.fusion_proj(torch.cat([hidden, meta], dim=-1))
        z, _ = self.sae(fused)  # reconstruction unused

        gamma, beta = self.film(z)
        coeffs_mod = gamma * coeffs_ns + beta

        forecast = self.symbolic_head(coeffs_mod)
        return forecast, coeffs_ns, coeffs_mod, z, gamma, beta
