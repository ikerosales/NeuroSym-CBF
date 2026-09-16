#!/usr/bin/env python
"""Train NeuroSym-CBF (multimodal model, MAE) and save the checkpoint.

Trains only -- it does not evaluate or export results/figures. Once you have the checkpoint, use:
    python scripts/10_evaluate.py --model neurosym-cbf

Usage:
    python scripts/08_train_neurosym_cbf.py
"""
from __future__ import annotations

import argparse
import logging

import torch
from torch.utils.data import DataLoader

from fred_forecast.config import data_dir, load_yaml_config, models_dir
from fred_forecast.datasets.load_series import load_metadata_embeddings, load_series_observations
from fred_forecast.datasets.timeseries_dataset import TimeSeriesDataset, split_train_test_obs
from fred_forecast.models.neurosym_cbf import NeuroSymCBFModel
from fred_forecast.training.train_neurosym_cbf import train_neurosym_cbf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()  # only for -h/--help

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
    df_train, _df_test = split_train_test_obs(df, d["context_len"], d["horizon_len"])
    logger.info("Training series: %d", len(df_train))

    train_set = TimeSeriesDataset(
        df_train,
        metadata_df,
        cntxt_length=d["context_len"],
        horizon_length=d["horizon_len"],
        padding_value=d["padding_value"],
        min_cntxt_length=d["min_context_len"],
        stage="train",
    )
    g = torch.Generator()
    g.manual_seed(config["seed"])
    num_workers = d.get("num_workers", 0)
    train_loader = DataLoader(
        train_set,
        batch_size=d["batch_size"],
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    logger.info("Train windows: %d", len(train_set))

    d_meta = metadata_df["embeddings"].iloc[0].shape[0]
    logger.info("Metadata embedding dim: %d", d_meta)

    model = NeuroSymCBFModel(
        seq_len=d["context_len"],
        horizon_len=d["horizon_len"],
        d_meta=d_meta,
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

    train_neurosym_cbf(model, train_loader, optimizer, scheduler, device, num_epochs=t["num_epochs"])

    checkpoint_path = models_dir() / "local" / f"neurosym_cbf_ctx{d['context_len']}_seed{config['seed']}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "seed": config["seed"],
            "context_len": d["context_len"],
            "horizon_len": d["horizon_len"],
            "d_model": b["d_model"],
            "seasonality": b["seasonality"],
            "dict_mult": sf["dict_mult"],
            "top_k": sf["top_k"],
            "d_meta": d_meta,
        },
        checkpoint_path,
    )
    logger.info("Checkpoint saved -> %s", checkpoint_path)


if __name__ == "__main__":
    main()
