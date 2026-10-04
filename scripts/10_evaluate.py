#!/usr/bin/env python
"""Evaluate a model (custom or baseline) on the test set and save the results.

One reusable evaluation script for ALL models (not one per model, as in
the original project). Saves aggregated metrics in JSON and per-sample arrays
in `.npz`, under `<RESULTS_DIR>/metrics/ctx{N}/{json,npz}/{model}.{json,npz}` -- see
`fred_forecast.evaluation.results_io`.

`tft` is the in-domain deep baseline: it is trained locally (`scripts/09_train_tft_baseline.py`), so
its default checkpoint is under `models/local/`.

Custom models (`unimodal-ablation`, `neurosym-cbf`): by default evaluate the checkpoint under
`models/imported_from_kaggle/` -- the original weights imported from Kaggle, which back every
number in the paper/report and which should ALWAYS be used for evaluating and publishing unless
told otherwise. Pass `--checkpoint` explicitly to evaluate a different one (e.g. a local retrain
from `scripts/07_*`/`08_*`). Baselines (`autoets-normafter`, `autoets-normbefore`,
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
    python scripts/10_evaluate.py --model unimodal-ablation --context-len 8 --d-model 148         --checkpoint models/local/unimodal_ablation_ctx8_seed40_d148.pt --label unimodal-ablation-wide
"""
from __future__ import annotations

import argparse
import logging

import torch
from torch.utils.data import DataLoader

from fred_forecast.config import data_dir, load_yaml_config, models_dir, results_dir
from fred_forecast.datasets.load_series import (
    METADATA_CONTROLS,
    apply_metadata_control,
    load_metadata_embeddings,
    load_series_observations,
)
from fred_forecast.datasets.timeseries_dataset import TimeSeriesDataset, split_train_test_obs
from fred_forecast.datasets.univariate_dataset import UnivariateTestDataset
from fred_forecast.evaluation.results_io import save_results
from fred_forecast.models.checkpoint_io import check_hyperparams, load_checkpoint
from fred_forecast.models.neurosym_cbf import NeuroSymCBFModel
from fred_forecast.models.tft_baseline import TFTBaselineModel
from fred_forecast.models.unimodal_ablation import UnimodalAblationModel
from fred_forecast.reproducibility import set_reproducible_environment
from fred_forecast.training.train_neurosym_cbf import evaluate_neurosym_cbf
from fred_forecast.training.train_unimodal_ablation import evaluate_unimodal_ablation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OWN_MODEL_CHOICES = ["unimodal-ablation", "neurosym-cbf", "tft"]
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


def _evaluate_own_model(
    model_name: str,
    checkpoint_arg: str | None,
    device: torch.device,
    context_len: int | None = None,
    d_model: int | None = None,
    metadata_control: str | None = None,
) -> dict:
    config = load_yaml_config("configs/model_protocol.yaml")
    d, b = config["data"], config["backbone"]
    if context_len is not None:
        d["context_len"] = context_len
    if d_model is not None:
        b["d_model"] = d_model
    database_json, embeddings_pkl = _database_paths()

    df = load_series_observations(database_json)
    metadata_df = load_metadata_embeddings(embeddings_pkl)
    if metadata_control:
        metadata_df = apply_metadata_control(metadata_df, metadata_control, config["seed"])
        logger.info("Metadata control: %s", metadata_control)
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
            models_dir() / "imported_from_kaggle" / f"unimodal_ablation_ctx{d['context_len']}_seed{config['seed']}.pt"
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
        state_dict, hyperparams = load_checkpoint(checkpoint_path, map_location=device)
        check_hyperparams(
            hyperparams,
            {"context_len": d["context_len"], "horizon_len": d["horizon_len"], "d_model": b["d_model"]},
        )
        model.load_state_dict(state_dict)
        results = evaluate_unimodal_ablation(
            model, test_loader, device, seasonality=b["seasonality"], padding_value=d["padding_value"]
        )
    elif model_name == "tft":
        tft = load_yaml_config("configs/baselines.yaml")["tft"]
        checkpoint_path = checkpoint_arg or (
            models_dir() / "local" / f"tft_ctx{d['context_len']}_seed{config['seed']}.pt"
        )
        model = TFTBaselineModel(
            seq_len=d["context_len"],
            horizon_len=d["horizon_len"],
            d_model=tft["d_model"],
            num_heads=tft["num_heads"],
            dropout=tft["dropout"],
            padding_value=d["padding_value"],
        ).to(device)
        state_dict, hyperparams = load_checkpoint(checkpoint_path, map_location=device)
        check_hyperparams(
            hyperparams,
            {"context_len": d["context_len"], "horizon_len": d["horizon_len"], "d_model": tft["d_model"]},
        )
        model.load_state_dict(state_dict)
        results = evaluate_unimodal_ablation(
            model, test_loader, device, seasonality=b["seasonality"], padding_value=d["padding_value"]
        )
    else:
        sf = config["sae_film"]
        checkpoint_path = checkpoint_arg or (
            models_dir() / "imported_from_kaggle" / f"neurosym_cbf_ctx{d['context_len']}_seed{config['seed']}.pt"
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
        state_dict, hyperparams = load_checkpoint(checkpoint_path, map_location=device)
        check_hyperparams(
            hyperparams,
            {"context_len": d["context_len"], "horizon_len": d["horizon_len"], "top_k": sf["top_k"]},
        )
        model.load_state_dict(state_dict)
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
    parser.add_argument(
        "--label", type=str, default=None,
        help="Name to save the results under (defaults to --model). Useful for evaluating an "
             "alternate checkpoint without overwriting the repo model's results",
    )
    parser.add_argument(
        "--context-len", type=int, default=None,
        help="Custom models only: overrides data.context_len from configs/model_protocol.yaml",
    )
    parser.add_argument(
        "--d-model", type=int, default=None,
        help="Custom models only: overrides backbone.d_model (e.g. 148 for the capacity-matched ablation)",
    )
    parser.add_argument(
        "--metadata-control", choices=METADATA_CONTROLS, default=None,
        help="neurosym-cbf only: evaluate with control embeddings (another series' or the mean one) "
             "instead of each series' own metadata",
    )
    args = parser.parse_args()

    protocol = load_yaml_config("configs/model_protocol.yaml")
    device = set_reproducible_environment(protocol["seed"], protocol["num_threads"])

    if args.model in OWN_MODEL_CHOICES:
        results, context_len = _evaluate_own_model(
            args.model, args.checkpoint, device,
            context_len=args.context_len, d_model=args.d_model, metadata_control=args.metadata_control,
        )
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

    label = args.label or args.model
    ctx_dir = results_dir() / "metrics" / f"ctx{context_len}"
    save_results(
        results,
        ctx_dir / "json" / f"{label}.json",
        ctx_dir / "npz" / f"{label}.npz",
        extra_metadata={"model": args.model, "label": label, "context_len": context_len},
    )


if __name__ == "__main__":
    main()
