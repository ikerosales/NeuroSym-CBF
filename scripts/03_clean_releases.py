#!/usr/bin/env python
"""Fase 2 del pipeline: limpieza y homogeneización temporal de los releases descargados.

Determina la ventana de contexto (por defecto 48 meses, ver configs/data.yaml:cleaning) que
maximiza el nº de series utilizables, descarta las que tienen demasiado NaN o NaN justo en el
borde de la ventana, y hace forward-fill del resto. Escribe el resultado bajo
`<DATA_DIR>/interim/final_dataset_<freq>/`.

Uso:
    python scripts/03_clean_releases.py
    python scripts/03_clean_releases.py --freq Monthly
"""
from __future__ import annotations

import argparse
import logging

from fred_forecast.config import data_dir, load_yaml_config
from fred_forecast.data.clean_releases import clean_releases

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_yaml_config("configs/data.yaml")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freq",
        default="Monthly",
        help="Frecuencia a limpiar. El TFG usa 'Monthly' (el valor por defecto).",
    )
    args = parser.parse_args()

    raw_dir = data_dir() / config["paths"]["raw_dir"] / f"parquet_{args.freq.strip().lower()}"
    output_dir = data_dir() / config["paths"]["interim_dir"] / f"final_dataset_{args.freq.strip().lower()}"

    logger.info("Limpiando %s -> %s", raw_dir, output_dir)
    clean_releases(
        parquet_dir=raw_dir,
        output_dir=output_dir,
        context_months=config["cleaning"]["context_months"],
        max_nan_ratio=config["cleaning"]["max_nan_ratio"],
    )


if __name__ == "__main__":
    main()
