"""Temporal Fusion Transformer baseline (Lim et al., 2021), trained on the same data as the custom models.

It is the in-domain deep baseline: unlike the pretrained foundation models, it is trained from scratch
on the same windows, with the same loss, optimizer and schedule as Unimodal-ablation / NeuroSym-CBF,
and it produces an unconstrained forecast instead of going through the symbolic head.

This is a reimplementation of the TFT building blocks for the setting of this project, not a port of
a library: there are no static covariates and no exogenous inputs, so the variable selection networks
and the static enrichment context of the original model have nothing to select or condition on. What
remains is the temporal backbone: gated residual networks, an LSTM encoder-decoder, the interpretable
multi-head attention over the whole window with a causal mask, and gated skip connections. The output
is a point forecast trained with MAE (the median of the original quantile output).

Inputs per time step: for the context, the normalized value and a flag marking left-padding; for the
horizon, only a learned position embedding (the "known future input" is the step index).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GatedLinearUnit(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_in, 2 * d_out)

    def forward(self, x):
        value, gate = self.fc(self.dropout(x)).chunk(2, dim=-1)
        return value * torch.sigmoid(gate)


class GateAddNorm(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.glu = GatedLinearUnit(d_model, d_model, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, skip):
        return self.norm(self.glu(x) + skip)


class GatedResidualNetwork(nn.Module):
    def __init__(self, d_in: int, d_model: int, dropout: float):
        super().__init__()
        self.skip = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()
        self.fc1 = nn.Linear(d_in, d_model)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(d_model, d_model)
        self.gate = GateAddNorm(d_model, dropout)

    def forward(self, x):
        return self.gate(self.fc2(self.elu(self.fc1(x))), self.skip(x))


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with value weights shared across heads and averaged head outputs."""

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q = nn.Linear(d_model, self.d_head * num_heads)
        self.k = nn.Linear(d_model, self.d_head * num_heads)
        self.v = nn.Linear(d_model, self.d_head)
        self.out = nn.Linear(self.d_head, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        batch, n_q, _ = query.shape
        n_k = key.shape[1]
        q = self.q(query).view(batch, n_q, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k(key).view(batch, n_k, self.num_heads, self.d_head).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / self.d_head**0.5
        scores = scores.masked_fill(mask, float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))  # (B, heads, n_q, n_k)
        v = self.v(value).unsqueeze(1)  # (B, 1, n_k, d_head), shared across heads
        return self.out((attn @ v).mean(dim=1))


class TFTBaselineModel(nn.Module):
    def __init__(self, seq_len: int, horizon_len: int, d_model: int, num_heads: int, dropout: float, padding_value: float):
        super().__init__()
        self.seq_len = seq_len
        self.horizon_len = horizon_len
        self.padding_value = padding_value

        self.past_input = GatedResidualNetwork(2, d_model, dropout)
        self.future_position = nn.Parameter(torch.zeros(horizon_len, d_model))
        nn.init.normal_(self.future_position, std=0.02)
        self.future_input = GatedResidualNetwork(d_model, d_model, dropout)

        self.encoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.decoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.post_lstm = GateAddNorm(d_model, dropout)

        self.enrichment = GatedResidualNetwork(d_model, d_model, dropout)
        self.attention = InterpretableMultiHeadAttention(d_model, num_heads, dropout)
        self.post_attention = GateAddNorm(d_model, dropout)
        self.positionwise = GatedResidualNetwork(d_model, d_model, dropout)
        self.pre_output = GateAddNorm(d_model, dropout)
        self.output = nn.Linear(d_model, 1)

        total = seq_len + horizon_len
        # Decoder position i (absolute index seq_len + i) may attend to every position up to itself.
        causal = torch.triu(torch.ones(total, total, dtype=torch.bool), diagonal=1)[seq_len:]
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, context):
        """`context`: (B, L, 1). Returns `(forecast (B, H, 1), None)`, the same pair as the custom models."""
        is_pad = (context == self.padding_value).to(context.dtype)
        past = self.past_input(torch.cat([context, is_pad], dim=-1))
        future = self.future_input(self.future_position).unsqueeze(0).expand(context.shape[0], -1, -1)

        encoded, state = self.encoder(past)
        decoded, _ = self.decoder(future, state)
        temporal = self.post_lstm(torch.cat([encoded, decoded], dim=1), torch.cat([past, future], dim=1))

        enriched = self.enrichment(temporal)
        attended = self.attention(enriched[:, self.seq_len :], enriched, enriched, self.causal_mask)
        attended = self.post_attention(attended, enriched[:, self.seq_len :])
        out = self.pre_output(self.positionwise(attended), temporal[:, self.seq_len :])
        return self.output(out), None
