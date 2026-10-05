#!/usr/bin/env python
"""Intervenciones en los metadatos de NeuroSym-CBF sobre un checkpoint ya entrenado.

Mide qué le pasa al forecast y a los conceptos del SAE cuando se le cambia a cada serie el
embedding de metadatos: ablación, permutación aleatoria (el nulo), swaps dirigidos entre tags y
control negativo entre series de tags idénticos. Ver `fred_forecast.interpretability.interventions`
para el detalle de cada nivel.

No re-entrena nada y no toca el contexto temporal: el encoder de la serie se ejecuta una sola vez
por batch y se reutiliza en todas las intervenciones.

Resultados en `<RESULTS_DIR>/interventions/metadata_interventions_ctx{N}.{json,npz}`.

Uso:
    python scripts/13_metadata_interventions.py
    python scripts/13_metadata_interventions.py --max-batches 2      # prueba rápida
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd
import torch

from fred_forecast.config import data_dir, load_yaml_config, models_dir, results_dir
from fred_forecast.datasets.load_series import load_metadata_embeddings, load_series_observations
from fred_forecast.datasets.timeseries_dataset import TimeSeriesDataset, split_train_test_obs
from fred_forecast.evaluation.results_io import save_results
from fred_forecast.interpretability.interventions import (
    ablation_interventions,
    baseline_intervention,
    random_permutation_intervention,
    run_metadata_interventions,
    tag_sets_for,
    tag_swap_with_matched_null,
    within_tagset_swap_intervention,
)
from fred_forecast.models.checkpoint_io import check_hyperparams, load_checkpoint
from fred_forecast.models.neurosym_cbf import NeuroSymCBFModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _build_test_set(context_len: int | None = None):
    data_config = load_yaml_config("configs/data.yaml")
    protocol = load_yaml_config("configs/model_protocol.yaml")
    d = protocol["data"]
    if context_len is not None:
        d["context_len"] = context_len

    context_months = data_config["cleaning"]["context_months"]
    database_json = (
        data_dir() / data_config["paths"]["processed_dir"] / f"fred_database_context_{context_months}.json"
    )
    embeddings_pkl = data_dir() / data_config["paths"]["embeddings_dir"] / "metadata_embeddings.pkl"

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
    return test_set, protocol, metadata_df, data_config


def _load_model(protocol: dict, d_meta: int, checkpoint_arg: str | None, device: torch.device):
    d, b, sf = protocol["data"], protocol["backbone"], protocol["sae_film"]
    checkpoint_path = checkpoint_arg or (
        models_dir() / f"neurosym_cbf_ctx{d['context_len']}_seed{protocol['seed']}.pt"
    )
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
    check_hyperparams(hyperparams, {"context_len": d["context_len"], "top_k": sf["top_k"]})
    model.load_state_dict(state_dict)
    logger.info("Checkpoint cargado <- %s", checkpoint_path)
    return model


def _build_interventions(config: dict, test_set, tags_df: pd.DataFrame) -> list:
    n_series = len(test_set.series_ids)
    interventions = [baseline_intervention(), *ablation_interventions()]

    for seed in config["permutation_seeds"]:
        interventions.append(random_permutation_intervention(n_series, seed))

    tag_sets = tag_sets_for(test_set.series_ids, tags_df)
    n_untagged = sum(1 for ts in tag_sets if not ts)
    if n_untagged:
        logger.warning("%d series sin tags: no entran en swaps dirigidos ni en el control", n_untagged)

    for pair in config["tag_swaps"]:
        swap, matched_null = tag_swap_with_matched_null(
            pair["a"], pair["b"], tag_sets, seed=config["seed"], max_pairs=pair.get("max_pairs")
        )
        if swap.n_affected == 0:
            logger.warning("Swap %r sin pares disponibles, se omite", swap.name)
            continue
        interventions.extend([swap, matched_null])

    if config["within_tagset_swap"]["enabled"]:
        interventions.append(
            within_tagset_swap_intervention(
                tag_sets, seed=config["seed"], min_group_size=config["within_tagset_swap"]["min_group_size"]
            )
        )

    for iv in interventions:
        logger.info("  %-32s %s", iv.name, iv.notes)
    return interventions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=None, help="Ruta al checkpoint de NeuroSym-CBF")
    parser.add_argument("--context-len", type=int, default=None, help="Sobrescribe data.context_len")
    parser.add_argument(
        "--label", type=str, default="metadata_interventions",
        help="Prefijo de los ficheros de salida (para no pisar los de otro checkpoint)",
    )
    parser.add_argument("--max-batches", type=int, default=None, help="Limita batches (prueba rápida)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    config = load_yaml_config("configs/interventions.yaml")
    test_set, protocol, metadata_df, data_config = _build_test_set(args.context_len)
    model = _load_model(protocol, metadata_df["embeddings"].iloc[0].shape[0], args.checkpoint, device)

    tags_df = pd.read_parquet(data_dir() / data_config["paths"]["tags_dir"] / "fred_tags_output.parquet")
    interventions = _build_interventions(config, test_set, tags_df)

    results = run_metadata_interventions(
        model,
        test_set,
        interventions,
        device,
        batch_size=protocol["data"]["batch_size"],
        max_batches=args.max_batches,
    )

    summary = results["interventions"]
    logger.info("%-32s %10s %12s %10s %10s", "intervención", "n", "ΔMAE", "% peor", "Jaccard")
    for name, row in summary.items():
        logger.info(
            "%-32s %10d %12.5f %9.1f%% %10.3f",
            name, row["n_evaluated"], row["delta_mae_mean"], 100 * row["frac_worse"], row["jaccard_mean"],
        )

    context_len = protocol["data"]["context_len"]
    out_dir = results_dir() / "interventions"
    save_results(
        results,
        out_dir / f"{args.label}_ctx{context_len}.json",
        out_dir / f"{args.label}_ctx{context_len}.npz",
        extra_metadata={
            "model": "neurosym-cbf",
            "context_len": context_len,
            "seed": protocol["seed"],
            "top_k": protocol["sae_film"]["top_k"],
            "dict_size": protocol["sae_film"]["dict_mult"] * protocol["backbone"]["d_model"],
        },
    )


if __name__ == "__main__":
    main()
