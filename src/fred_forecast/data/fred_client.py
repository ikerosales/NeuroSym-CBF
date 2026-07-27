"""Cliente ligero para la API REST de FRED (Federal Reserve Economic Data).

Cubre las dos llamadas que usa el pipeline de descarga masiva: listar todos los release IDs
(API v1) y descargar, paginadas, las observaciones de un release filtradas por frecuencia
(API v2).

Incluye `units` por defecto: aunque la última versión del driver de descarga original lo
desactivaba (`include_units=False`), al inspeccionar la base de datos v4 real que se usó para
todo el TFG, el campo `units` está poblado en prácticamente el 100% de las series muestreadas
(los releases ya existían en disco de una descarga anterior con `include_units=True` y el
driver se limitaba a saltarlos) — y además se usa en el texto de los embeddings de metadatos
("Measured in {units}", ver la Fase 4). `notes` sí se excluye:
en la base de datos real está poblado de forma inconsistente según cuándo se descargó cada
release, y no lo usa ninguna fase posterior — la información de notas relevante (de tags,
no de series) se obtiene aparte, ver `download_tags.py`.
"""
from __future__ import annotations

import logging
import time
from typing import Iterator

import pandas as pd
import requests

logger = logging.getLogger(__name__)

RELEASES_URL = "https://api.stlouisfed.org/fred/releases"
RELEASE_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/v2/release/observations"

VALID_FREQUENCIES = [
    "Daily",
    "Weekly",
    "Biweekly",
    "Monthly",
    "Quarterly",
    "Semiannual",
    "Annual",
    "5 Year Frequency",
]


def get_all_release_ids(api_key: str, page_size: int = 1000) -> list[int]:
    """Devuelve todos los release IDs disponibles en FRED (paginado)."""
    params: dict = {"api_key": api_key, "file_type": "json", "limit": page_size, "offset": 0}
    releases: list[dict] = []

    while True:
        response = requests.get(RELEASES_URL, params=params)
        response.raise_for_status()
        batch = response.json().get("releases", [])
        releases.extend(batch)

        if len(batch) < page_size:
            break
        params["offset"] += page_size

    release_ids = [r["id"] for r in releases]
    logger.info("Encontrados %d releases", len(release_ids))
    return release_ids


def iter_release_observations(
    release_id: int,
    api_key: str,
    freq: str,
    page_limit: int = 500_000,
    rate_limit_wait_seconds: int = 20,
    include_units: bool = True,
    include_notes: bool = False,
) -> Iterator[pd.DataFrame]:
    """Itera las páginas de observaciones de `release_id` filtradas a `freq`.

    Cada fila es una observación (series_id, date, value, title, y opcionalmente units/notes).
    Reintenta indefinidamente ante HTTP 429 (rate limit) esperando `rate_limit_wait_seconds`;
    para cualquier otro error HTTP corta la descarga de ese release (se registra y se continúa
    con el siguiente release desde el llamador).
    """
    if freq not in VALID_FREQUENCIES:
        raise ValueError(f"Frecuencia {freq!r} no válida. Usa una de {VALID_FREQUENCIES}.")

    headers = {"Authorization": f"Bearer {api_key}"}
    params: dict = {"release_id": release_id, "format": "json", "limit": page_limit}
    has_more = True

    while has_more:
        response = requests.get(RELEASE_OBSERVATIONS_URL, params=params, headers=headers)

        if response.status_code == 429:
            logger.warning("Rate limit en release %s, esperando %ds", release_id, rate_limit_wait_seconds)
            time.sleep(rate_limit_wait_seconds)
            continue
        if response.status_code != 200:
            logger.error("HTTP %d descargando release %s", response.status_code, release_id)
            return

        data = response.json()
        rows = []
        for series in data.get("series", []):
            if series.get("frequency") != freq:
                continue
            for obs in series.get("observations", []):
                val = obs.get("value")
                row = {
                    "series_id": series["series_id"],
                    "date": obs["date"],
                    "value": float(val) if val != "." else None,
                    "title": series.get("title"),
                }
                if include_units:
                    row["units"] = series.get("units")
                if include_notes:
                    row["notes"] = series.get("notes")
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            yield df

        has_more = data.get("has_more", False)
        if has_more:
            params["next_cursor"] = data.get("next_cursor")
