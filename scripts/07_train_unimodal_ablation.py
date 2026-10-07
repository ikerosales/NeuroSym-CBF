#!/usr/bin/env python
"""Train Unimodal-ablation (ablation baseline without metadata) and save the checkpoint.

Trains only -- it does not evaluate or export results/figures. Once you have the checkpoint, use:
    python scripts/10_evaluate.py --model unimodal-ablation

The protocol values come from `configs/model_protocol.yaml`. Two of them can be overridden from the
command line without editing the config: `--context-len` (to sweep contexts) and `--d-model` (to
train the capacity-matched control: with `--d-model 148` the ablation has about as many trainable
parameters as NeuroSym-CBF, instead of roughly a fifth). A non-default width is recorded in the
checkpoint name, so it never overwrites the standard ablation.

Usage:
    python scripts/07_train_unimodal_ablation.py
    python scripts/07_train_unimodal_ablation.py --context-len 8 --d-model 148
"""
from __future__ import annotations

import argparse
import logging

import torch
from torch.utils.data import DataLoader

from fred_forecast.config import data_dir, load_yaml_config, models_dir
from fred_forecast.datasets.load_series import load_metadata_embeddings, load_series_observations
from fred_forecast.datasets.timeseries_dataset import TimeSeriesDataset, split_train_test_obs
from fred_forecast.models.unimodal_ablation import UnimodalAblationModel
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
    parser.add_argument("--d-model", type=int, default=None, help="Overrides backbone.d_model")
    args = parser.parse_args()

    config = load_yaml_config("configs/model_protocol.yaml")
    data_config = load_yaml_config("configs/data.yaml")
    d, b, t = config["data"], config["backbone"], config["training"]
    context_months = data_config["cleaning"]["context_months"]

    if args.context_len is not None:
        d["context_len"] = args.context_len
        d["min_context_len"] = min(d["min_context_len"], args.context_len)
    if args.min_context_len is not None:
        d["min_context_len"] = args.min_context_len
    width_tag = ""
    if args.d_model is not None and args.d_model != b["d_model"]:
        b["d_model"] = args.d_model
        width_tag = f"_d{args.d_model}"

    device = set_reproducible_environment(config["seed"], config["num_threads"])
    logger.info(
        "context_len: %d | min_context_len: %d | d_model: %d",
        d["context_len"], d["min_context_len"], b["d_model"],
    )

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

    model = UnimodalAblationModel(
        seq_len=d["context_len"],
        horizon_len=d["horizon_len"],
        patch_len=b["patch_len"],
        patch_stride=b["patch_stride"],
        d_model=b["d_model"],
        d_ff_ts=b["d_ff_ts"],
        dropout=b["dropout"],
        seasonality=b["seasonality"],
        num_blocks=b["num_blocks"],
    ).to(device)
    logger.info("Trainable parameters: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["scheduler_t_max"])

    train_unimodal_ablation(model, train_loader, optimizer, scheduler, device, num_epochs=t["num_epochs"])

    checkpoint_path = models_dir() / "local" / f"unimodal_ablation_ctx{d['context_len']}_seed{config['seed']}{width_tag}.pt"
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
            "num_blocks": b["num_blocks"],
            "seasonality": b["seasonality"],
            "dropout": b["dropout"],
        },
        checkpoint_path,
    )
    logger.info("Checkpoint saved -> %s", checkpoint_path)


if __name__ == "__main__":
    main()
