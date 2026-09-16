#!/usr/bin/env python
"""Names the atoms of NeuroSym-CBF's SAE dictionary via FRED tag enrichment.

Uses the `z` activations already saved by `scripts/10_evaluate.py` (does not re-evaluate the
model) and aligns them with `series_id` by reconstructing the same deterministic split. For each
atom of the dictionary, computes which FRED tags appear disproportionately among the series that
activate it (one-tailed hypergeometric test) -- see `interpretability/concept_naming.py`.

Usage:
    python scripts/15_concept_naming.py --npz results/metrics/neurosym-cbf-kaggle_ctx30.npz \\
        --context-len 30
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from fred_forecast.config import data_dir, load_yaml_config, results_dir
from fred_forecast.interpretability.concept_naming import (
    load_tag_sets,
    name_all_concepts,
    reconstruct_test_series_order,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", type=Path, required=True, help="Evaluation results with a 'z' (SAE) array")
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--horizon-len", type=int, default=None, help="Defaults to the one in configs/model_protocol.yaml")
    parser.add_argument("--database", type=Path, default=None, help="Defaults to the original (in-sample) database")
    parser.add_argument("--tags-parquet", type=Path, default=None, help="Defaults to data/raw/tags/fred_tags_output.parquet")
    parser.add_argument("--min-tag-count", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--label", type=str, default=None, help="Output name; defaults to derive it from --npz")
    args = parser.parse_args()

    protocol = load_yaml_config("configs/model_protocol.yaml")
    data_config = load_yaml_config("configs/data.yaml")
    horizon_len = args.horizon_len or protocol["data"]["horizon_len"]

    database_json = args.database or (
        data_dir() / data_config["paths"]["processed_dir"] / f"fred_database_context_{data_config['cleaning']['context_months']}.json"
    )
    tags_parquet = args.tags_parquet or (data_dir() / data_config["paths"]["tags_dir"] / "fred_tags_output.parquet")

    logger.info("z <- %s", args.npz)
    z = np.load(args.npz, allow_pickle=True)["z"]
    logger.info("z: %s", z.shape)

    logger.info("Reconstructing test set series_id order <- %s", database_json)
    series_ids = reconstruct_test_series_order(database_json, args.context_len, horizon_len)
    if len(series_ids) != len(z):
        raise SystemExit(
            f"Misaligned: {len(series_ids)} reconstructed series_id vs {len(z)} rows in 'z'. "
            "Check --context-len/--horizon-len/--database: they must match the evaluation run "
            "that generated the .npz."
        )

    logger.info("Tags <- %s", tags_parquet)
    tag_sets = load_tag_sets(tags_parquet)
    n_with_tags = sum(1 for sid in series_ids if sid in tag_sets)
    logger.info("Series with tags: %d / %d", n_with_tags, len(series_ids))

    results = name_all_concepts(
        z, series_ids, tag_sets, min_tag_count_in_active=args.min_tag_count, top_n=args.top_n
    )

    n_dead = sum(1 for r in results if r["n_active"] == 0)
    n_named = sum(1 for r in results if r["top_tags"])
    n_selective = sum(1 for r in results if r["top_tags"] and r["top_tags"][0]["lift"] >= 5)
    logger.info(
        "Atoms: %d total | %d never activate | %d with a candidate tag | %d with lift>=5 "
        "(confidently nameable, not just 'almost always active')",
        len(results), n_dead, n_named, n_selective,
    )

    label = args.label or args.npz.stem
    output_dir = results_dir() / "concepts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"concept_naming_{label}.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "npz": str(args.npz), "context_len": args.context_len, "horizon_len": horizon_len,
                "n_series": len(series_ids), "n_with_tags": n_with_tags, "dict_size": z.shape[1],
                "n_dead_atoms": n_dead, "n_named_atoms": n_named, "atoms": results,
            },
            f, indent=2,
        )
    logger.info("Saved to %s", output_file)

    # Sorted by lift, not by n_active: with tens of thousands of samples the p-value saturates to
    # ~0 even for a lift of 1.04 (an atom that's almost always active), so sorting by p-value alone
    # would only surface generic atoms. Lift is the signal of how *specific* the atom is to that
    # tag, which is what's needed to name it.
    named = [r for r in results if r["top_tags"]]
    print(f"\n--- Top 15 MOST SPECIFIC atoms (highest lift of the best candidate tag) ---")
    for r in sorted(named, key=lambda r: -r["top_tags"][0]["lift"])[:15]:
        t = r["top_tags"][0]
        print(f"  atom {r['atom']:3d}  n_active={r['n_active']:6d}  '{t['tag']}'  lift={t['lift']:6.1f}  "
              f"({t['n_in_active']}/{r['n_active']} = {t['active_rate']*100:.0f}% vs base {t['base_rate']*100:.2f}%)")

    print(f"\n--- Top 15 MOST GENERIC atoms (highest n_active, almost always active) ---")
    for r in sorted(results, key=lambda r: -r["n_active"])[:15]:
        best = r["top_tags"][0] if r["top_tags"] else None
        tag_txt = f"'{best['tag']}' lift={best['lift']:.2f}" if best else "(no candidate tag)"
        print(f"  atom {r['atom']:3d}  n_active={r['n_active']:6d}  {tag_txt}")


if __name__ == "__main__":
    main()
