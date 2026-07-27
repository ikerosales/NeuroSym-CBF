#!/usr/bin/env python
"""Entrena y evalúa NeuroSym-CBF (pinball) — trabajo preliminar/futuro, ver
`experimental/README.md`. A diferencia del core no se ha separado train/evaluate en dos
scripts (no es el pipeline principal, no vale la pena esa inversión todavía), pero sí
reutiliza el mismo formato de resultados que el core: JSON de métricas agregadas + .npz de
arrays por muestra, bajo `<RESULTS_DIR>/metrics/`.

Ejecutar desde la raíz del repo:
    python experimental/train_neurosym_cbf_pinball.py
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
from fred_forecast.datasets.load_series import load_metadata_embeddings, load_series_observations
from fred_forecast.datasets.timeseries_dataset import TimeSeriesDataset, split_train_test_obs
from fred_forecast.evaluation.results_io import save_results

from neurosym_cbf_pinball.model import NeuroSymCBFQuantileModel
from neurosym_cbf_pinball.train import evaluate_neurosym_cbf_pinball, train_neurosym_cbf_pinball

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QUANTILES = [0.5, 0.9]
MEDIAN_IDX = QUANTILES.index(0.5)
UPPER_IDX = QUANTILES.index(0.9)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()  # solo para -h/--help

    config = load_yaml_config("configs/model_protocol.yaml")
    data_config = load_yaml_config("configs/data.yaml")
    d, b, sf, t = config["data"], config["backbone"], config["sae_film"], config["training"]
    context_months = data_config["cleaning"]["context_months"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])
    logger.info("Device: %s", device)

    database_json = (
        data_dir() / data_config["paths"]["processed_dir"] / f"fred_database_context_{context_months}.json"
    )
    embeddings_pkl = data_dir() / data_config["paths"]["embeddings_dir"] / "metadata_embeddings.pkl"

    df = load_series_observations(database_json)
    metadata_df = load_metadata_embeddings(embeddings_pkl)
    df_train, df_test = split_train_test_obs(df, d["context_len"], d["horizon_len"])

    common_kwargs = dict(
        df_metadata=metadata_df,
        cntxt_length=d["context_len"],
        horizon_length=d["horizon_len"],
        padding_value=d["padding_value"],
        min_cntxt_length=d["min_context_len"],
    )
    train_set = TimeSeriesDataset(df_train, stage="train", **common_kwargs)
    test_set = TimeSeriesDataset(df_test, stage="test", **common_kwargs)

    g = torch.Generator()
    g.manual_seed(config["seed"])
    train_loader = DataLoader(train_set, batch_size=d["batch_size"], shuffle=True, generator=g)
    test_loader = DataLoader(test_set, batch_size=d["batch_size"], shuffle=False)
    logger.info("Train windows: %d | Test windows: %d", len(train_set), len(test_set))

    d_meta = metadata_df["embeddings"].iloc[0].shape[0]
    quantiles = torch.tensor(QUANTILES)

    model = NeuroSymCBFQuantileModel(
        seq_len=d["context_len"],
        horizon_len=d["horizon_len"],
        d_meta=d_meta,
        num_quantiles=len(QUANTILES),
        patch_len=b["patch_len"],
        patch_stride=b["patch_stride"],
        d_model=b["d_model"],
        d_ff_ts=b["d_ff_ts"],
        dropout=b["dropout"],
        seasonality=b["seasonality"],
        num_blocks=b["num_blocks"],
        dict_mult=sf["dict_mult"],
        top_k=sf["top_k"],
        d_ff_film=sf["d_ff_meta"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["scheduler_t_max"])

    train_neurosym_cbf_pinball(
        model, train_loader, optimizer, scheduler, device, quantiles, num_epochs=t["num_epochs"]
    )
    results = evaluate_neurosym_cbf_pinball(
        model,
        test_loader,
        device,
        quantiles,
        median_idx=MEDIAN_IDX,
        upper_idx=UPPER_IDX,
        seasonality=b["seasonality"],
        padding_value=d["padding_value"],
    )
    logger.info(
        "Pinball mean: %.4f | Coverage 90: %.4f | Interval width: %.4f",
        results["pinball_mean"], results["coverage_90_mean"], results["interval_width_mean"],
    )

    metrics_dir = results_dir() / "metrics"
    save_results(
        results,
        metrics_dir / f"neurosym-cbf-pinball_ctx{d['context_len']}.json",
        metrics_dir / f"neurosym-cbf-pinball_ctx{d['context_len']}.npz",
        extra_metadata={
            "model": "neurosym-cbf-pinball",
            "context_len": d["context_len"],
            "horizon_len": d["horizon_len"],
            "seed": config["seed"],
            "quantiles": QUANTILES,
        },
    )


if __name__ == "__main__":
    main()
