"""Fase 3 del pipeline: empaqueta los parquets limpios (Fase 2) en una única base de datos JSON.

Un registro por serie: título, observaciones, fecha de inicio, longitud, etc. Es la estructura
que consume el resto del pipeline (embeddings de metadatos, entrenamiento, evaluación). El
campo `tags` se deja siempre a `None` aquí — los tags se descargan y usan por separado (ver
`download_tags.py` y, más adelante, `embeddings/text_encoder.py`), no se fusionan en este JSON.

Nota: los parquets de Fase 1/2 se descargan con `include_units=False, include_notes=False` (ver
`fred_client.py`), así que en la práctica `units` y `notes` siempre acaban en `None` — se dejan
como campos condicionales por si en el futuro se reactiva su descarga.
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

import pandas as pd
import tqdm

logger = logging.getLogger(__name__)


def build_series_record(series_id: str, group: pd.DataFrame) -> dict:
    """Construye el registro de una serie a partir de sus observaciones ya limpias."""
    return {
        "title": group["title"].iloc[0] if "title" in group.columns else None,
        "series_id": series_id,
        "observations": group["value"].tolist(),
        "start_date": group["date"].min().strftime("%Y-%m-%d"),
        "units": group["units"].iloc[0] if "units" in group.columns else None,
        "length": len(group),
        "tags": None,  # los tags se descargan/usan por separado, ver download_tags.py
        "notes": group["notes"].iloc[0] if "notes" in group.columns else None,
    }


def build_database(processed_dir: Path, context_months: int) -> list[dict]:
    """Recorre todos los parquets limpios de `processed_dir` y construye la base de datos completa.

    El primer elemento de la lista devuelta es un objeto de cabecera con metadata global
    (`context_length`, `release_date`); el resto son registros por serie (ver `build_series_record`).
    """
    header = {
        "context_length": context_months,
        "release_date": datetime.datetime.now().strftime("%Y-%m-%d"),
    }
    database: list[dict] = [header]

    for release_parquet in tqdm.tqdm(sorted(processed_dir.glob("*.parquet")), desc="construyendo BD"):
        df = pd.read_parquet(release_parquet, engine="fastparquet")
        for series_id, group in df.groupby("series_id"):
            database.append(build_series_record(series_id, group))

    logger.info("Base de datos construida: %d series", len(database) - 1)
    return database


def save_database(database: list[dict], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(database, f, indent=4)
    logger.info("Guardado en %s", output_file)
