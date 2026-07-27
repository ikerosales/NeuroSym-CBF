#!/usr/bin/env python
"""Utilidad LOCAL de conveniencia: consolida el dataset FRED completo en un solo parquet/jsonl.

ATENCION: el resultado de este script NUNCA debe publicarse ni commitearse — son las
observaciones completas de FRED redistribuidas en otro formato, lo que las condiciones de uso
de FRED no permiten. Solo pensado como atajo de trabajo local (un fichero en vez de 114
parquets sueltos). Escribe bajo `<DATA_DIR>/processed/`, que ya está gitignorado por completo.

No es un paso numerado del pipeline: es opcional y no lo usa ningún otro script.

Uso:
    python scripts/export_local_dataset.py
"""
from __future__ import annotations

import argparse
import logging

from fred_forecast.config import data_dir, load_yaml_config
from fred_forecast.data.local_dataset_export import build_consolidated_dataset, save_jsonl_gz, save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()  # solo para -h/--help
    logger.warning(
        "Este export consolida el dataset FRED COMPLETO (observaciones incluidas) en un solo "
        "fichero. NO lo publiques ni lo commitees (las condiciones de uso de FRED no lo permiten)."
    )

    config = load_yaml_config("configs/data.yaml")
    context_months = config["cleaning"]["context_months"]

    database_json = data_dir() / config["paths"]["processed_dir"] / f"fred_database_context_{context_months}.json"
    tags_dir = data_dir() / config["paths"]["tags_dir"]
    output_dir = data_dir() / config["paths"]["processed_dir"]

    df = build_consolidated_dataset(
        database_json_path=database_json,
        tags_parquet_path=tags_dir / "fred_tags_output.parquet",
        notes_lookup_path=tags_dir / "tags_notes_lookup.json",
    )
    save_parquet(df, output_dir / "fred_dataset_consolidated.parquet")
    save_jsonl_gz(df, output_dir / "fred_dataset_consolidated.jsonl.gz")


if __name__ == "__main__":
    main()
