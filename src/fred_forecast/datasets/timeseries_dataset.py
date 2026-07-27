"""Dataset compartido por las tres fases del modelo propio (Fase 6).

Ventana deslizante (train) o ventana única (test), normalización min-max por ventana,
padding por la izquierda, y clip del horizonte. Código idéntico en las tres variantes de
origen del modelo (Unimodal-ablation, NeuroSym-CBF, NeuroSym-CBF pinball), factorizado en un único
módulo para no triplicarlo, sin tocar ni una línea de la lógica.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def split_train_test_obs(df: pd.DataFrame, cntxt_length: int, horizon_length: int):
    """Split determinista: las últimas `horizon_length` observaciones van a test, el resto
    (excluyendo ese horizonte) a train. Descarta series demasiado cortas para tener al
    menos `cntxt_length + horizon_length` observaciones.
    """
    df = df[df["observations"].apply(lambda x: len(x) >= cntxt_length + horizon_length)]
    df = df.copy()
    df["train_obs"] = df["observations"].apply(lambda x: x[:-horizon_length])
    df["test_obs"] = df["observations"].apply(lambda x: x[-(cntxt_length + horizon_length):])
    df_train = df[["series_id", "train_obs"]].rename(columns={"train_obs": "observations"})
    df_test = df[["series_id", "test_obs"]].rename(columns={"test_obs": "observations"})
    return df_train, df_test


class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset (train) or single-window dataset (test).
    Returns:
        context  (L, 1)
        horizon  (H, 1)
        mask     (H, 1)
        metadata (D_meta,)
        stats    (2,)  = [c_min, denom] for later denormalisation
    """

    def __init__(
        self,
        df,
        df_metadata,
        cntxt_length,
        horizon_length,
        padding_value,
        min_cntxt_length,
        stage,
        sampling="sliding",
    ):
        self.observations = [np.asarray(x, dtype=np.float32) for x in df["observations"]]
        self.cntxt_length = cntxt_length
        self.horizon_length = horizon_length
        self.padding_value = padding_value
        self.min_cntxt_length = min_cntxt_length
        self.lengths = [len(ts) for ts in self.observations]
        self.series_ids = list(df["series_id"])

        self.metadata = np.stack(
            [df_metadata.loc[sid]["embeddings"] for sid in self.series_ids]
        )  # (N_series, D_meta)

        if stage == "train" and sampling == "sliding":
            ts_idx_list, t_idx_list = [], []
            for i, length in enumerate(self.lengths):
                ts = np.arange(self.min_cntxt_length, length - horizon_length + 1, dtype=np.uint32)
                ts_idx_list.append(np.full(len(ts), i, dtype=np.uint32))
                t_idx_list.append(ts)
            self._ts_idx = np.concatenate(ts_idx_list)
            self._t_idx = np.concatenate(t_idx_list)
        elif stage == "test":
            self._ts_idx = np.arange(len(self.lengths), dtype=np.uint32)
            self._t_idx = np.array([l - horizon_length for l in self.lengths], dtype=np.uint32)

    def __len__(self):
        return len(self._ts_idx)

    def __getitem__(self, idx):
        ts_idx = int(self._ts_idx[idx])
        t = int(self._t_idx[idx])

        ts = self.observations[ts_idx]
        metadata = self.metadata[ts_idx]

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
            torch.tensor(metadata, dtype=torch.float32),
            torch.tensor([c_min, denom], dtype=torch.float32),
        )
