#!/usr/bin/env python
"""Evaluate a model (custom or baseline) on the test set and save the results.

One reusable evaluation script for ALL models (not one per model, as in
the original project). Saves aggregated metrics in JSON and per-sample arrays
in `.npz`, under `<RESULTS_DIR>/metrics/{model}_ctx{N}.{json,npz}` -- see
`fred_forecast.evaluation.results_io`.

Custom models (`unimodal-ablation`, `neurosym-cbf`): evaluate an already trained checkpoint
from `scripts/07_*`/`08_*`. Baselines (`autoets-normafter`, `autoets-normbefore`,
`chronos-bolt`, `chronos-2`, `timesfm`): zero-shot, no custom checkpoint required.

The context length (`context_len`) for each baseline is read from `configs/baselines.yaml` -- in the thesis
AutoETS/Chronos-Bolt were evaluated with ctx=8 and Chronos-2/TimesFM with ctx=30 (a real
inconsistency in the original protocol); change it there if you want to rerun any of them with another context.

Usage:
    python scripts/10_evaluate.py --model unimodal-ablation
    python scripts/10_evaluate.py --model neurosym-cbf
    python scripts/10_evaluate.py --model autoets-normafter
    python scripts/10_evaluate.py --model autoets-normbefore
    python scripts/10_evaluate.py --model chronos-bolt
    python scripts/10_evaluate.py --model chronos-2
    python scripts/10_evaluate.py --model timesfm
"""
from __future__ import annotations

import argparse
import logging

import torch
from torch.utils.data import DataLoader

from fred_forecast.config import data_dir, load_yaml_config, models_dir, results_dir
from fred_forecast.datasets.load_series import load_metadata_embeddings, load_series_observations
from fred_forecast.datasets.timeseries_dataset import TimeSeriesDataset, split_train_test_obs
from fred_forecast.datasets.univariate_dataset import UnivariateTestDataset
from fred_forecast.evaluation.results_io import save_results
from fred_forecast.models.neurosym_cbf import NeuroSymCBFModel
from fred_forecast.models.unimodal_ablation import UnimodalAblationModel
from fred_forecast.training.train_neurosym_cbf import evaluate_neurosym_cbf
from fred_forecast.training.train_unimodal_ablation import evaluate_unimodal_ablation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OWN_MODEL_CHOICES = ["unimodal-ablation", "neurosym-cbf"]
BASELINE_CHOICES = ["autoets-normafter", "autoets-normbefore", "chronos-bolt", "chronos-2", "timesfm"]
MODEL_CHOICES = OWN_MODEL_CHOICES + BASELINE_CHOICES


def _database_paths():
    data_config = load_yaml_config("configs/data.yaml")
    context_months = data_config["cleaning"]["context_months"]
    database_json = (
        data_dir() / data_config["paths"]["processed_dir"] / f"fred_database_context_{context_months}.json"
    )
    embeddings_pkl = data_dir() / data_config["paths"]["embeddings_dir"] / "metadata_embeddings.pkl"
    return database_json, embeddings_pkl


def _evaluate_own_model(model_name: str, checkpoint_arg: str | None, device: torch.device) -> dict:
    config = load_yaml_config("configs/model_protocol.yaml")
    d, b = config["data"], config["backbone"]
    database_json, embeddings_pkl = _database_paths()

    df = load_series_observations(database_json)
    metadata_df = load_metadata_embeddings(embeddings_pkl)
    _df_train, df_test = split_train_test_obs(df, d["context_len"], d["horizon_len"])
    test_set = TimeSeriesDataset(
        df_test,
        metadata_df,
        cntxt_length=d["context_len"],
        horizon_length=d["horizon_len"],
        padding_value=d["padding_value"],
        min_cntxt_length=d["min_context_len"],
        stage="test",
    )
    logger.info("Test windows: %d", len(test_set))
    test_loader = DataLoader(test_set, batch_size=d["batch_size"], shuffle=False)

    if model_name == "unimodal-ablation":
        checkpoint_path = checkpoint_arg or (
            models_dir() / f"unimodal_ablation_ctx{d['context_len']}_seed{config['seed']}.pt"
        )
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
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        results = evaluate_unimodal_ablation(
            model, test_loader, device, seasonality=b["seasonality"], padding_value=d["padding_value"]
        )
    else:
        sf = config["sae_film"]
        checkpoint_path = checkpoint_arg or (
            models_dir() / f"neurosym_cbf_ctx{d['context_len']}_seed{config['seed']}.pt"
        )
        d_meta = metadata_df["embeddings"].iloc[0].shape[0]
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
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        results = evaluate_neurosym_cbf(
            model, test_loader, device, seasonality=b["seasonality"], padding_value=d["padding_value"]
        )

    logger.info("Checkpoint loaded <- %s", checkpoint_path)
    return results, d["context_len"]


def _evaluate_autoets(variant: str) -> dict:
    from fred_forecast.evaluation.baselines.autoets import evaluate_autoets

    baselines_config = load_yaml_config("configs/baselines.yaml")
    cfg = baselines_config["autoets"]
    database_json, _embeddings_pkl = _database_paths()

    df = load_series_observations(database_json)
    _df_train, df_test = split_train_test_obs(df, cfg["context_len"], baselines_config["horizon_len"])
    logger.info("Test series: %d", len(df_test))

    results = evaluate_autoets(
        df_test,
        variant=variant,
        cntxt_len=cfg["context_len"],
        horizon_len=baselines_config["horizon_len"],
        seasonality=baselines_config["seasonality"],
        n_jobs=cfg["n_jobs"],
    )
    return results, cfg["context_len"]


def _build_univariate_loader(cfg: dict, baselines_config: dict) -> DataLoader:
    database_json, _embeddings_pkl = _database_paths()
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
    return DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False)


def _evaluate_chronos_bolt(device: torch.device) -> dict:
    from chronos import BaseChronosPipeline

    from fred_forecast.evaluation.baselines.chronos import evaluate_chronos_bolt

    baselines_config = load_yaml_config("configs/baselines.yaml")
    cfg = baselines_config["chronos_bolt"]
    loader = _build_univariate_loader(cfg, baselines_config)

    pipeline = BaseChronosPipeline.from_pretrained(
        cfg["model_id"], device_map=str(device), torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    pipeline.model.eval()

    results = evaluate_chronos_bolt(
        pipeline, loader, device, seasonality=baselines_config["seasonality"], padding_value=baselines_config["padding_value"]
    )
    return results, cfg["context_len"]


def _evaluate_chronos2(device: torch.device) -> dict:
    from chronos import BaseChronosPipeline

    from fred_forecast.evaluation.baselines.chronos import evaluate_chronos2

    baselines_config = load_yaml_config("configs/baselines.yaml")
    cfg = baselines_config["chronos2"]
    loader = _build_univariate_loader(cfg, baselines_config)

    pipeline = BaseChronosPipeline.from_pretrained(cfg["model_id"], device_map=str(device))

    results = evaluate_chronos2(
        pipeline, loader, device, seasonality=baselines_config["seasonality"], padding_value=baselines_config["padding_value"]
    )
    return results, cfg["context_len"]


def _evaluate_timesfm(device: torch.device) -> dict:
    import timesfm

    from fred_forecast.evaluation.baselines.timesfm import evaluate_timesfm

    baselines_config = load_yaml_config("configs/baselines.yaml")
    cfg = baselines_config["timesfm"]
    loader = _build_univariate_loader(cfg, baselines_config)

    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu" if device.type == "cuda" else "cpu",
            per_core_batch_size=cfg["batch_size"],
            horizon_len=baselines_config["horizon_len"],
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=cfg["checkpoint"]),
    )

    results = evaluate_timesfm(
        model,
        loader,
        device,
        horizon_len=baselines_config["horizon_len"],
        seasonality=baselines_config["seasonality"],
        padding_value=baselines_config["padding_value"],
    )
    return results, cfg["context_len"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to the checkpoint (custom models only); by default it is derived from --model"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    if args.model in OWN_MODEL_CHOICES:
        results, context_len = _evaluate_own_model(args.model, args.checkpoint, device)
    elif args.model in ("autoets-normafter", "autoets-normbefore"):
        variant = args.model.split("-", 1)[1]
        results, context_len = _evaluate_autoets(variant)
    elif args.model == "chronos-bolt":
        results, context_len = _evaluate_chronos_bolt(device)
    elif args.model == "chronos-2":
        results, context_len = _evaluate_chronos2(device)
    else:  # timesfm
        results, context_len = _evaluate_timesfm(device)

    logger.info(
        "MAE mean: %.4f | sMAPE mean: %.4f | MASE mean: %.4f",
        results["mae_mean"], results["smape_mean"], results["mase_mean"],
    )

    metrics_dir = results_dir() / "metrics"
    json_path = metrics_dir / f"{args.model}_ctx{context_len}.json"
    npz_path = metrics_dir / f"{args.model}_ctx{context_len}.npz"
    save_results(
        results,
        json_path,
        npz_path,
        extra_metadata={"model": args.model, "context_len": context_len},
    )


if __name__ == "__main__":
    main()
