"""Utilidad LOCAL de conveniencia: consolida la base de datos + tags en un único parquet/jsonl.

NO PUBLICAR NUNCA el resultado de este módulo. A diferencia del resto de `src/fred_forecast`,
esto no es parte del pipeline reproducible que se documenta/publica del TFG — es un atajo para
tener el dataset completo (observaciones incluidas) en un solo fichero en vez de 114 parquets
sueltos, útil solo para trabajo local. El resultado son las observaciones de FRED completas, que
las condiciones de uso de FRED no permiten redistribuir (ver README).
"""
from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fred_forecast.embeddings.text_encoder import load_tags_with_expanded_notes

logger = logging.getLogger(__name__)

PARQUET_SCHEMA = pa.schema(
    [
        ("series_id", pa.string()),
        ("title", pa.string()),
        ("observations", pa.list_(pa.float32())),
        ("units", pa.dictionary(pa.int32(), pa.string())),
        ("notes", pa.string()),
        ("start_date", pa.timestamp("ns")),
        ("tags", pa.list_(pa.string())),
        ("frequency", pa.dictionary(pa.int32(), pa.string())),
    ]
)


def _load_series_full(database_json_path: Path) -> pd.DataFrame:
    """Carga todos los campos de la base de datos (incluidas las observaciones)."""
    with open(database_json_path) as f:
        data = json.load(f)

    series = [
        {
            "series_id": x["series_id"],
            "title": x["title"],
            "observations": x["observations"],
            "units": x["units"],
            "notes": x.get("notes"),
            "start_date": x["start_date"],
        }
        for x in data
        if "series_id" in x
    ]
    return pd.DataFrame(series)


def build_consolidated_dataset(
    database_json_path: Path,
    tags_parquet_path: Path,
    notes_lookup_path: Path,
    frequency_label: str = "monthly",
) -> pd.DataFrame:
    """Fusiona la base de datos (Fase 3) con los tags expandidos (Fase 1) en un DataFrame tipado,
    listo para exportar a parquet/jsonl con `save_parquet`/`save_jsonl_gz`.
    """
    df = _load_series_full(database_json_path)
    df_tags = load_tags_with_expanded_notes(tags_parquet_path, notes_lookup_path)
    df = df.merge(df_tags[["series_id", "tags_mapped"]], on="series_id", how="left")
    df["frequency"] = frequency_label

    df["start_date"] = pd.to_datetime(df["start_date"])
    df["tags_mapped"] = df["tags_mapped"].apply(
        lambda x: [t.strip() for t in x.split(",")] if isinstance(x, str) else []
    )
    df["notes"] = df["notes"].fillna("")
    df["observations"] = df["observations"].apply(lambda lst: [float(x) for x in lst])
    df["frequency"] = df["frequency"].astype("category")
    df["units"] = df["units"].astype("category")

    return df.rename(columns={"tags_mapped": "tags"})


def save_parquet(df: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)
    pq.write_table(table, output_file, compression="zstd", compression_level=7)
    logger.info("Guardado %s", output_file)


def save_jsonl_gz(df: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["start_date"] = df["start_date"].astype(str)  # JSON no acepta timestamps
    with gzip.open(output_file, "wt", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record) + "\n")
    logger.info("Guardado %s", output_file)
