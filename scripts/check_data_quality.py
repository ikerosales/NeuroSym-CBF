#!/usr/bin/env python
"""Informe de calidad de datos para un directorio de parquets de releases descargados.

No es un paso numerado del pipeline (es una utilidad de diagnóstico ad hoc, ejecutable en
cualquier momento tras la Fase 1) — comprueba series_id duplicados entre releases, la
distribución de fechas de última actualización, y el % de NaN por serie.

Uso:
    python scripts/check_data_quality.py --parquet-dir data/raw/parquet_monthly
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fred_forecast.data.quality_checks import (
    find_duplicate_series_ids,
    last_update_stats,
    nan_percentage_by_series,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    args = parser.parse_args()

    duplicates = find_duplicate_series_ids(args.parquet_dir)
    logger.info("series_id duplicados entre releases: %d", len(duplicates))
    for sid, count in list(duplicates.items())[:10]:
        logger.info("  %s aparece %d veces", sid, count)

    last_dates = last_update_stats(args.parquet_dir)
    logger.info("Fecha de última actualización más antigua: %s", last_dates.iloc[0])
    logger.info("Percentil 5 de fechas de última actualización: %s", last_dates.iloc[int(0.05 * len(last_dates))])
    logger.info("Percentil 10 de fechas de última actualización: %s", last_dates.iloc[int(0.10 * len(last_dates))])

    nan_pct = nan_percentage_by_series(args.parquet_dir)
    logger.info(
        "%% NaN por serie — describe():\n%s",
        nan_pct.describe(percentiles=[0.5, 0.75, 0.85, 0.9, 0.95, 0.99]),
    )


if __name__ == "__main__":
    main()
