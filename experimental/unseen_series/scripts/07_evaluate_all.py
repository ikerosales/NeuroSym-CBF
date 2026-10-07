#!/usr/bin/env python
"""Unseen series, step 7: evaluate every model on the three sets, without retraining.

Calls `scripts/10_evaluate.py` (and `scripts/evaluate_moirai.py`, in its own environment) once per
model, set and context length, with `--database`/`--embeddings` pointing at this experiment's data
and `RESULTS_DIR` at `experimental/unseen_series/results`, so nothing is written into the main
`results/`. The trained models are the ones behind the paper's numbers
(`models/imported_from_kaggle/`).

Results: `results/metrics/ctx{L}/{json,npz}/{model}-{set}.{json,npz}`.

Usage:
    python experimental/unseen_series/scripts/07_evaluate_all.py
    python experimental/unseen_series/scripts/07_evaluate_all.py --models tft moirai --sets daily
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

from paths import CONTEXT_LENS, REPO_ROOT, RESULTS_DIR, SETS, database_path, embeddings_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS = ["neurosym-cbf", "unimodal-ablation", "tft", "autoets-normafter", "chronos-2", "chronos-bolt", "timesfm", "moirai"]


def moirai_python() -> str:
    """Interpreter of the separate Moirai environment (see `scripts/evaluate_moirai.py`).

    Its location inside `.venv-moirai/` depends on the platform and on how the environment was
    created (conda or venv), so the usual places are tried in turn.
    """
    env_dir = REPO_ROOT / ".venv-moirai"
    for candidate in ("python.exe", "Scripts/python.exe", "bin/python"):
        if (env_dir / candidate).exists():
            return str(env_dir / candidate)
    raise FileNotFoundError(f"No Python interpreter found in {env_dir}; see scripts/evaluate_moirai.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    parser.add_argument("--context-lens", nargs="+", type=int, default=list(CONTEXT_LENS))
    args = parser.parse_args()

    env = dict(os.environ)
    env["RESULTS_DIR"] = str(RESULTS_DIR)

    failures = []
    for model in args.models:
        for set_name in args.sets:
            for context_len in args.context_lens:
                label = f"{model}-{set_name}"
                database = database_path(set_name, context_len)
                if model == "moirai":
                    cmd = [
                        moirai_python(), "scripts/evaluate_moirai.py", "--context-len", str(context_len),
                        "--database", str(database), "--label", label,
                    ]
                else:
                    cmd = [
                        sys.executable, "scripts/10_evaluate.py", "--model", model, "--context-len", str(context_len),
                        "--database", str(database), "--embeddings", str(embeddings_path(set_name)), "--label", label,
                    ]
                logger.info("=== %s ctx%d ===", label, context_len)
                if subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode != 0:
                    logger.error("FAILED: %s ctx%d", label, context_len)
                    failures.append(f"{label}-ctx{context_len}")

    if failures:
        raise SystemExit(f"{len(failures)} evaluations failed: {failures}")


if __name__ == "__main__":
    main()
