#!/usr/bin/env python
"""Fase 3 del pipeline: empaqueta los parquets limpios (Fase 2) en la base de datos JSON final.

Uso:
    python scripts/04_build_database.py
    python scripts/04_build_database.py --freq Monthly
"""
from __future__ import annotations

import argparse
import logging

from fred_forecast.config import data_dir, load_yaml_config
from fred_forecast.data.build_database import build_database, save_database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_yaml_config("configs/data.yaml")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freq",
        default="Monthly",
        help="Frecuencia cuyo dataset limpio (Fase 2) se empaqueta en la base de datos.",
    )
    args = parser.parse_args()

    context_months = config["cleaning"]["context_months"]
    processed_input_dir = data_dir() / config["paths"]["interim_dir"] / f"final_dataset_{args.freq.strip().lower()}"
    output_file = (
        data_dir() / config["paths"]["processed_dir"] / f"fred_database_context_{context_months}.json"
    )

    logger.info("Empaquetando %s -> %s", processed_input_dir, output_file)
    database = build_database(processed_input_dir, context_months=context_months)
    save_database(database, output_file)


if __name__ == "__main__":
    main()
