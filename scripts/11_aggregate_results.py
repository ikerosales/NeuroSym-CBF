#!/usr/bin/env python
"""Agrega los resultados de todos los modelos evaluados en tablas comparativas.

Lee `<RESULTS_DIR>/metrics/*.json` (modelos propios + baselines, generados por
`scripts/10_evaluate.py`) y escribe una tabla comparativa por familia de métricas:
- `<RESULTS_DIR>/comparison_point.csv` — forecast puntual (MAE/sMAPE/MASE): Unimodal-ablation,
  NeuroSym-CBF, AutoETS, Chronos-Bolt, Chronos-2, TimesFM.
- `<RESULTS_DIR>/comparison_probabilistic.csv` — pinball/cobertura/anchura: solo si hay
  resultados de NeuroSym-CBF (pinball) / Chronos-2 (pinball) (`experimental/`, ver README).

Uso:
    python scripts/11_aggregate_results.py
"""
from __future__ import annotations

import argparse
import logging

from fred_forecast.config import results_dir
from fred_forecast.evaluation.aggregate_results import build_comparison_tables, load_all_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()  # solo para -h/--help

    metrics_dir = results_dir() / "metrics"
    results = load_all_results(metrics_dir)
    logger.info("Resultados encontrados en %s: %d", metrics_dir, len(results))

    point_df, prob_df = build_comparison_tables(results)
    out_dir = results_dir()

    if not point_df.empty:
        point_path = out_dir / "comparison_point.csv"
        point_df.to_csv(point_path, index=False)
        logger.info("Comparativa (forecast puntual) -> %s", point_path)
        print(point_df.to_string(index=False))
    else:
        logger.warning("No hay resultados de forecast puntual (campo 'mae_mean') todavía.")

    if not prob_df.empty:
        prob_path = out_dir / "comparison_probabilistic.csv"
        prob_df.to_csv(prob_path, index=False)
        logger.info("Comparativa (probabilística) -> %s", prob_path)
        print(prob_df.to_string(index=False))
    else:
        logger.info("Sin resultados probabilísticos (NeuroSym-CBF/Chronos-2 pinball son trabajo experimental, ver experimental/README.md).")


if __name__ == "__main__":
    main()
