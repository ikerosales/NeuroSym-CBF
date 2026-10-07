#!/usr/bin/env python
"""Unseen series, step 5: download the FRED tags of the series of one set.

Runs the main `scripts/02_download_tags.py` on the union of the set's series (over the context
lengths), with `DATA_DIR` pointing at the set's folder and the set name as file prefix, so the main
tag download is not touched. Needs `FRED_API_KEY` in `.env`.

Usage:
    python experimental/unseen_series/scripts/05_download_tags.py --set weekly
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys

from paths import CONTEXT_LENS, REPO_ROOT, SETS, database_path, set_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=list(SETS), required=True)
    args = parser.parse_args()

    ids: set[str] = set()
    for database in {database_path(args.set, c) for c in CONTEXT_LENS}:
        ids.update(json.loads(database.with_name(database.stem + "_series_ids.json").read_text()))
    union_file = set_dir(args.set) / "processed" / f"{args.set}_series_ids_union.json"
    union_file.write_text(json.dumps(sorted(ids)))
    logger.info("%d series_id -> %s", len(ids), union_file)

    env = dict(os.environ)
    env["DATA_DIR"] = str(set_dir(args.set))
    cmd = [sys.executable, "scripts/02_download_tags.py", "--series-ids-file", str(union_file), "--prefix", args.set]
    logger.info("Running: %s (DATA_DIR=%s)", " ".join(cmd), env["DATA_DIR"])
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
