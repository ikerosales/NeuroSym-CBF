#!/usr/bin/env python
"""Train the Temporal Fusion Transformer baseline on the same data as the custom models.

In-domain deep baseline: same training windows, loss (masked MAE), optimizer, schedule and number of
epochs as Unimodal-ablation / NeuroSym-CBF (`configs/model_protocol.yaml`); only the model changes
(`configs/baselines.yaml`, block `tft`). Trains only -- once you have the checkpoint, use:
    python scripts/10_evaluate.py --model tft

A TFT step is far more expensive than a step of the custom models (an LSTM over the whole window plus
attention). `--max-batches` caps the number of batches per epoch for a budgeted run; the cap is
recorded in the checkpoint name so a partial run is never mistaken for the full protocol.

Usage:
    python scripts/09_train_tft_baseline.py --context-len 30
    python scripts/09_train_tft_baseline.py --context-len 8 --max-batches 2000
"""
from __future__ import annotations

import argparse
import logging

import torch
from torch.utils.data import DataLoader

from fred_forecast.config import data_dir, load_yaml_config, models_dir
from fred_forecast.datasets.load_series import load_metadata_embeddings, load_series_observations
from fred_forecast.datasets.timeseries_dataset import TimeSeriesDataset, split_train_test_obs
from fred_forecast.models.tft_baseline import TFTBaselineModel
from fred_forecast.reproducibility import set_reproducible_environment
from fred_forecast.training.train_unimodal_ablation import train_unimodal_ablation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context-len", type=int, default=None, help="Overrides data.context_len")
    parser.add_argument(
        "--min-context-len", type=int, default=None,
        help="Overrides data.min_context_len (default with --context-len: min(config value, context_len))",
    )
    parser.add_argument("--max-batches", type=int, default=None, help="Cap on batches per epoch (budgeted run)")
    args = parser.parse_args()

    config = load_yaml_config("configs/model_protocol.yaml")
    data_config = load_yaml_config("configs/data.yaml")
    tft = load_yaml_config("configs/baselines.yaml")["tft"]
    d, t = config["data"], config["training"]
    context_months = data_config["cleaning"]["context_months"]

    if args.context_len is not None:
        d["context_len"] = args.context_len
        d["min_context_len"] = min(d["min_context_len"], args.context_len)
    if args.min_context_len is not None:
        d["min_context_len"] = args.min_context_len
    budget_tag = f"_max{args.max_batches}" if args.max_batches is not None else ""

    device = set_reproducible_environment(config["seed"], config["num_threads"])
    logger.info(
        "context_len: %d | min_context_len: %d | d_model: %d | max_batches: %s",
        d["context_len"], d["min_context_len"], tft["d_model"], args.max_batches,
    )

    database_json = (
        data_dir() / data_config["paths"]["processed_dir"] / f"fred_database_context_{context_months}.json"
    )
    embeddings_pkl = data_dir() / data_config["paths"]["embeddings_dir"] / "metadata_embeddings.pkl"

    df = load_series_observations(database_json)
    # The TFT does not use the metadata; the embeddings are loaded only because TimeSeriesDataset
    # returns them, which keeps the batches identical to the ones the custom models train on.
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

    model = TFTBaselineModel(
        seq_len=d["context_len"],
        horizon_len=d["horizon_len"],
        d_model=tft["d_model"],
        num_heads=tft["num_heads"],
        dropout=tft["dropout"],
        padding_value=d["padding_value"],
    ).to(device)
    logger.info("Trainable parameters: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["scheduler_t_max"])

    train_unimodal_ablation(
        model, train_loader, optimizer, scheduler, device, num_epochs=t["num_epochs"], max_batches=args.max_batches
    )

    checkpoint_path = models_dir() / "local" / f"tft_ctx{d['context_len']}_seed{config['seed']}{budget_tag}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "seed": config["seed"],
            "context_len": d["context_len"],
            "horizon_len": d["horizon_len"],
            "d_model": tft["d_model"],
            "num_heads": tft["num_heads"],
            "dropout": tft["dropout"],
            "max_batches": args.max_batches,
        },
        checkpoint_path,
    )
    logger.info("Checkpoint saved -> %s", checkpoint_path)


if __name__ == "__main__":
    main()
