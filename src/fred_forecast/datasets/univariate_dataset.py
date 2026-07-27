"""Dataset de test, sin metadatos, para baselines zero-shot (Chronos-Bolt, Chronos-2, TimesFM).

Estos modelos son univariantes (no consumen embeddings de metadatos), así que no necesitan
la rama de metadatos ni el muestreo por ventana deslizante de
`fred_forecast.datasets.timeseries_dataset.TimeSeriesDataset` — solo una ventana de test por
serie. Es una versión reducida de ese dataset (misma normalización min-max por ventana y
padding, sin la rama de metadatos), compartida por los tres baselines de foundation models.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class UnivariateTestDataset(Dataset):
    """
    Una única ventana de test por serie (últimas `horizon_length` observaciones).
    Returns:
        context (L, 1), horizon (H, 1), mask (H, 1), stats (2,) = [c_min, denom]
    """

    def __init__(self, df, cntxt_length, horizon_length, padding_value, min_cntxt_length):
        self.observations = [np.asarray(x, dtype=np.float32) for x in df["observations"]]
        self.cntxt_length = cntxt_length
        self.horizon_length = horizon_length
        self.padding_value = padding_value
        self.min_cntxt_length = min_cntxt_length
        self.lengths = [len(ts) for ts in self.observations]

        self._ts_idx = np.arange(len(self.lengths), dtype=np.uint32)
        self._t_idx = np.array([l - horizon_length for l in self.lengths], dtype=np.uint32)

    def __len__(self):
        return len(self._ts_idx)

    def __getitem__(self, idx):
        ts_idx = int(self._ts_idx[idx])
        t = int(self._t_idx[idx])

        ts = self.observations[ts_idx]
        context = ts[max(0, t - self.cntxt_length) : t]
        horizon = ts[t : t + self.horizon_length]

        c_min = context.min()
        c_max = context.max()
        denom = c_max - c_min if c_max != c_min else 1e-5
        context = (context - c_min) / denom
        horizon = (horizon - c_min) / denom

        clip_value = 8.0
        horizon = np.clip(horizon, -clip_value + 1.0, clip_value)

        if len(context) < self.cntxt_length:
            pad = np.full(self.cntxt_length - len(context), self.padding_value, dtype=np.float32)
            context = np.concatenate([pad, context])

        mask = np.ones(self.horizon_length, dtype=bool)
        if len(horizon) < self.horizon_length:
            pad_len = self.horizon_length - len(horizon)
            horizon = np.concatenate([horizon, np.zeros(pad_len, dtype=np.float32)])
            mask[-pad_len:] = False

        return (
            torch.from_numpy(context).unsqueeze(-1),
            torch.from_numpy(horizon).unsqueeze(-1),
            torch.from_numpy(mask).unsqueeze(-1),
            torch.tensor([c_min, denom], dtype=torch.float32),
        )
