"""Carga de series y embeddings de metadatos para entrenamiento/evaluación del modelo propio.

Lee la base de datos construida en la Fase 3 (`build_database.py`, JSON) y los embeddings de
metadatos de la Fase 4 (`text_encoder.py`, pickle) — solo los campos que necesita
`TimeSeriesDataset`: `series_id`/`observations` y `series_id`/`embeddings` respectivamente.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_series_observations(database_json_path: Path) -> pd.DataFrame:
    """DataFrame con columnas `series_id`, `observations` (np.float32 arrays), a partir de
    la base de datos JSON generada por `build_database.py`.
    """
    with open(database_json_path) as f:
        data = json.load(f)

    rows = [
        {"series_id": str(x["series_id"]), "observations": np.asarray(x["observations"], dtype=np.float32)}
        for x in data
        if "series_id" in x
    ]
    return pd.DataFrame(rows)


def load_metadata_embeddings(embeddings_pkl_path: Path) -> pd.DataFrame:
    """DataFrame indexado por `series_id`, columna `embeddings` (np.float32 arrays), a
    partir del pickle generado por `text_encoder.build_metadata_embeddings`.
    """
    metadata_df = pd.read_pickle(embeddings_pkl_path)
    metadata_df["series_id"] = metadata_df["series_id"].astype(str)
    metadata_df["embeddings"] = metadata_df["embeddings"].apply(lambda x: np.asarray(x, dtype=np.float32))
    return metadata_df.set_index("series_id")
