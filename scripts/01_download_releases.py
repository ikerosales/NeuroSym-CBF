#!/usr/bin/env python
"""Fase 1 del pipeline: descarga masiva de observaciones FRED por release.

Descarga, para cada frecuencia configurada en configs/data.yaml (Monthly en el TFG), las
observaciones de TODOS los releases de FRED y las guarda incrementalmente en parquet bajo
`<DATA_DIR>/raw/parquet_<freq>/release_<id>.parquet`.

Uso:
    python scripts/01_download_releases.py
    python scripts/01_download_releases.py --freq Monthly
"""
from __future__ import annotations

import argparse
import logging

from fred_forecast.config import data_dir, get_env, load_yaml_config
from fred_forecast.data.download_releases import download_all_releases

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_yaml_config("configs/data.yaml")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freq",
        choices=config["frequencies"],
        help="Si se omite, descarga todas las frecuencias listadas en configs/data.yaml",
    )
    args = parser.parse_args()

    api_key = get_env("FRED_API_KEY")
    freqs = [args.freq] if args.freq else config["frequencies"]
    raw_dir = data_dir() / config["paths"]["raw_dir"]

    for freq in freqs:
        output_dir = raw_dir / f"parquet_{freq.strip().lower()}"
        logger.info("Descargando releases con frecuencia %s -> %s", freq, output_dir)
        download_all_releases(freq=freq, output_dir=output_dir, api_key=api_key)


if __name__ == "__main__":
    main()
