#!/usr/bin/env python
"""Evalúa Chronos-2 en modo probabilístico (pinball) — trabajo preliminar/futuro, ver
`experimental/README.md`. Baseline zero-shot pensado para compararse contra
`experimental/train_neurosym_cbf_pinball.py`, no contra los modelos core.

Ejecutar desde la raíz del repo:
    python experimental/evaluate_chronos2_pinball.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader

from fred_forecast.config import data_dir, load_yaml_config, results_dir
from fred_forecast.datasets.load_series import load_series_observations
from fred_forecast.datasets.timeseries_dataset import split_train_test_obs
from fred_forecast.datasets.univariate_dataset import UnivariateTestDataset
from fred_forecast.evaluation.results_io import save_results

from chronos2_pinball.evaluate import evaluate_chronos2_pinball

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QUANTILES = [0.5, 0.9]


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()  # solo para -h/--help

    from chronos import BaseChronosPipeline

    data_config = load_yaml_config("configs/data.yaml")
    baselines_config = load_yaml_config("configs/baselines.yaml")
    cfg = baselines_config["chronos2"]  # mismo contexto que el baseline puntual (ctx=30)
    context_months = data_config["cleaning"]["context_months"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    database_json = (
        data_dir() / data_config["paths"]["processed_dir"] / f"fred_database_context_{context_months}.json"
    )
    df = load_series_observations(database_json)
    _df_train, df_test = split_train_test_obs(df, cfg["context_len"], baselines_config["horizon_len"])
    test_set = UnivariateTestDataset(
        df_test,
        cntxt_length=cfg["context_len"],
        horizon_length=baselines_config["horizon_len"],
        padding_value=baselines_config["padding_value"],
        min_cntxt_length=cfg["min_context_len"],
    )
    logger.info("Test windows: %d", len(test_set))
    loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False)

    pipeline = BaseChronosPipeline.from_pretrained(cfg["model_id"], device_map=str(device))
    quantiles = torch.tensor(QUANTILES)

    results = evaluate_chronos2_pinball(
        pipeline,
        loader,
        device,
        horizon_len=baselines_config["horizon_len"],
        quantiles=quantiles,
        seasonality=baselines_config["seasonality"],
        padding_value=baselines_config["padding_value"],
    )
    logger.info(
        "Pinball q50: %.4f | q90: %.4f | Coverage 90: %.4f | Cruces de cuantil: %.2f%%",
        results["pinball_q50_mean"], results["pinball_q90_mean"],
        results["coverage_90_mean"], 100 * results["quantile_crossing_frac"],
    )

    metrics_dir = results_dir() / "metrics"
    save_results(
        results,
        metrics_dir / f"chronos2-pinball_ctx{cfg['context_len']}.json",
        metrics_dir / f"chronos2-pinball_ctx{cfg['context_len']}.npz",
        extra_metadata={
            "model": "chronos2-pinball",
            "context_len": cfg["context_len"],
            "horizon_len": baselines_config["horizon_len"],
            "quantiles": QUANTILES,
        },
    )


if __name__ == "__main__":
    main()
