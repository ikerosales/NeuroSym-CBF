#!/usr/bin/env python
"""Evaluate the Moirai foundation model (Salesforce, `uni2ts`) on the test set, without fine-tuning.

Unnumbered on purpose: it does not run in the project's environment. `uni2ts` pins older versions of
torch and numpy than this package requires, so it lives in its own environment and this script imports
the few pieces of `fred_forecast` it needs straight from `src/` instead of the installed package:

    conda create -p .venv-moirai python=3.11 pip
    .venv-moirai/python -m pip install uni2ts
    .venv-moirai/python scripts/evaluate_moirai.py --context-len 30

Same test windows, normalization and metrics as the other baselines (`UnivariateTestDataset`,
`forecasting_metrics`), and the same output layout as `scripts/10_evaluate.py`:
`<RESULTS_DIR>/metrics/ctx{N}/{json,npz}/moirai.{json,npz}`.

Moirai is probabilistic: the point forecast is the median of `--num-samples` sample paths, like the
median used for Chronos. The patch size is fixed to 8, the value the Moirai paper assigns to monthly
data that fits the short contexts used here.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule  # noqa: E402

from fred_forecast.config import data_dir, load_yaml_config, results_dir  # noqa: E402
from fred_forecast.datasets.load_series import load_series_observations  # noqa: E402
from fred_forecast.datasets.timeseries_dataset import split_train_test_obs  # noqa: E402
from fred_forecast.datasets.univariate_dataset import UnivariateTestDataset  # noqa: E402
from fred_forecast.evaluation.results_io import save_results  # noqa: E402
from fred_forecast.metrics.forecasting_metrics import (  # noqa: E402
    masked_mae_per_sample,
    masked_mase_per_sample,
    masked_smape_per_sample,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_moirai(model, loader, device, seasonality, padding_value, num_samples, max_batches=None):
    all_mae, all_smape, all_mase = [], [], []
    all_preds, all_targets, all_contexts = [], [], []
    started = time.time()

    for batch_idx, (context, horizon, mask, stats) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        context = context.to(device)  # (B, L, 1)
        horizon = horizon.to(device)
        mask_f = mask.to(device, dtype=torch.float32)

        is_pad = context.squeeze(-1) == padding_value  # (B, L)
        samples = model(
            past_target=context,
            past_observed_target=~is_pad.unsqueeze(-1),
            past_is_pad=is_pad,
            num_samples=num_samples,
        )  # (B, num_samples, H) for a single target dimension
        pred = samples.reshape(samples.shape[0], samples.shape[1], -1).median(dim=1).values.unsqueeze(-1)

        all_mae.extend(masked_mae_per_sample(pred, horizon, mask_f).cpu().numpy())
        all_smape.extend(masked_smape_per_sample(pred, horizon, mask_f).cpu().numpy())
        all_mase.extend(
            masked_mase_per_sample(
                pred, horizon, context, mask_f, seasonality=seasonality, padding_value=padding_value
            )
            .cpu()
            .numpy()
        )
        all_preds.append(pred.cpu())
        all_targets.append(horizon.cpu())
        all_contexts.append(context.cpu())

        if batch_idx % 25 == 0:
            logger.info(
                "Batch [%d/%d] | %.1f s/batch | running MAE: %.4f",
                batch_idx + 1, len(loader), (time.time() - started) / (batch_idx + 1), float(np.mean(all_mae)),
            )

    mae_arr, smape_arr, mase_arr = np.array(all_mae), np.array(all_smape), np.array(all_mase)
    return {
        "mae_mean": float(mae_arr.mean()),
        "mae_median": float(np.median(mae_arr)),
        "mae_std": float(mae_arr.std()),
        "mae_p25": float(np.percentile(mae_arr, 25)),
        "mae_p75": float(np.percentile(mae_arr, 75)),
        "mae_p90": float(np.percentile(mae_arr, 90)),
        "smape_mean": float(smape_arr.mean()),
        "smape_median": float(np.median(smape_arr)),
        "mase_mean": float(mase_arr.mean()),
        "mase_median": float(np.median(mase_arr)),
        "mae_per_sample": mae_arr,
        "smape_per_sample": smape_arr,
        "mase_per_sample": mase_arr,
        "preds": torch.cat(all_preds),
        "targets": torch.cat(all_targets),
        "contexts": torch.cat(all_contexts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--model-id", default="Salesforce/moirai-1.1-R-base")
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-batches", type=int, default=None, help="Stop early (timing / smoke test)")
    parser.add_argument("--label", default="moirai", help="Name to save the results under")
    parser.add_argument(
        "--database", type=str, default=None,
        help="Series database (JSON, same format as the main one) to evaluate on instead of the main database",
    )
    args = parser.parse_args()

    protocol = load_yaml_config("configs/model_protocol.yaml")
    baselines = load_yaml_config("configs/baselines.yaml")
    data_config = load_yaml_config("configs/data.yaml")
    horizon_len, seasonality, padding_value = (
        baselines["horizon_len"], baselines["seasonality"], baselines["padding_value"],
    )
    min_context_len = min(protocol["data"]["min_context_len"], args.context_len)

    torch.set_num_threads(protocol["num_threads"])
    np.random.seed(protocol["seed"])
    torch.manual_seed(protocol["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | model: %s | context_len: %d | samples: %d", device, args.model_id, args.context_len, args.num_samples)

    database_json = (
        data_dir() / data_config["paths"]["processed_dir"]
        / f"fred_database_context_{data_config['cleaning']['context_months']}.json"
    )
    if args.database:
        database_json = Path(args.database)
    df = load_series_observations(database_json)
    _df_train, df_test = split_train_test_obs(df, args.context_len, horizon_len)
    del df
    test_set = UnivariateTestDataset(
        df_test,
        cntxt_length=args.context_len,
        horizon_length=horizon_len,
        padding_value=padding_value,
        min_cntxt_length=min_context_len,
    )
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    logger.info("Test windows: %d", len(test_set))

    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(args.model_id),
        prediction_length=horizon_len,
        context_length=args.context_len,
        patch_size=args.patch_size,
        num_samples=args.num_samples,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    ).to(device)
    model.eval()

    results = evaluate_moirai(
        model, loader, device, seasonality, padding_value, args.num_samples, max_batches=args.max_batches
    )
    logger.info(
        "MAE mean: %.4f | sMAPE mean: %.4f | MASE mean: %.4f",
        results["mae_mean"], results["smape_mean"], results["mase_mean"],
    )

    if args.max_batches is not None:
        logger.info("--max-batches set: partial run, results not saved")
        return
    ctx_dir = results_dir() / "metrics" / f"ctx{args.context_len}"
    save_results(
        results,
        ctx_dir / "json" / f"{args.label}.json",
        ctx_dir / "npz" / f"{args.label}.npz",
        extra_metadata={
            "model": "moirai", "label": args.label, "context_len": args.context_len,
            "model_id": args.model_id, "patch_size": args.patch_size, "num_samples": args.num_samples,
            "seed": protocol["seed"],
        },
    )


if __name__ == "__main__":
    main()
