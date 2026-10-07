#!/usr/bin/env python
"""Unseen series, step 6: T5 metadata embeddings for the series of one set.

Runs the main `scripts/05_build_metadata_embeddings.py` once per database of the set and keeps the
union: the text (title, tags, units) does not depend on the context length, so one pickle serves
every context.

Tag notes: a tag is expanded to its descriptive note before encoding. For the monthly set the
lookup of the training corpus is merged in with priority, so a tag the model already knows is
expanded exactly as in training. The weekly and daily sets use only their own lookup; that is how
the embeddings behind the reported numbers were built, and it is kept so they stay reproducible.

Usage:
    python experimental/unseen_series/scripts/06_build_embeddings.py --set weekly
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys

import pandas as pd
from paths import CONTEXT_LENS, REPO_ROOT, SETS, database_path, embeddings_path, tags_dir

from fred_forecast.config import data_dir, load_yaml_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=list(SETS), required=True)
    args = parser.parse_args()

    set_tags = tags_dir(args.set)
    lookups = [set_tags / f"{args.set}_notes_lookup.json"]
    if args.set == "monthly":
        main_tags = data_dir() / load_yaml_config("configs/data.yaml")["paths"]["tags_dir"]
        lookups.insert(0, main_tags / "tags_notes_lookup.json")

    output_file = embeddings_path(args.set)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Longest context first: where a series is in several databases, that row is the one kept.
    databases = list(dict.fromkeys(database_path(args.set, c) for c in sorted(CONTEXT_LENS, reverse=True)))
    parts = []
    for i, database in enumerate(databases):
        part_file = output_file.with_name(f"_part_{i}.pkl")
        cmd = [
            sys.executable, "scripts/05_build_metadata_embeddings.py",
            "--database", str(database),
            "--tags-parquet", str(set_tags / f"{args.set}_output.parquet"),
            "--notes-lookup", *[str(p) for p in lookups],
            "--output", str(part_file),
        ]
        logger.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        parts.append(pd.read_pickle(part_file))
        part_file.unlink()

    merged = pd.concat(parts, ignore_index=True).drop_duplicates("series_id", keep="first")
    merged.to_pickle(output_file)
    logger.info("%s embeddings: %d series -> %s", args.set, len(merged), output_file)


if __name__ == "__main__":
    main()
