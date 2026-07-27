#!/usr/bin/env python
"""Fase 1 del pipeline (metadatos): descarga los tags de cada series_id desde FRED.

Lee la lista canónica de series desde `series_ids/series_ids.json` (las 194.442 series que
componen la base de datos final v4) y descarga sus tags de forma async, con checkpointing
para poder reanudar si se corta.

Uso:
    python scripts/02_download_tags.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from fred_forecast.config import data_dir, get_env, load_yaml_config
from fred_forecast.data.download_tags import download_all_tags

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()  # solo para -h/--help
    config = load_yaml_config("configs/data.yaml")
    api_key = get_env("FRED_API_KEY")

    series_ids_file = Path(config["paths"]["series_ids_file"])
    series_ids = json.loads(series_ids_file.read_text())
    logger.info("Cargadas %d series desde %s", len(series_ids), series_ids_file)

    tags_dir = data_dir() / config["paths"]["tags_dir"]
    asyncio.run(
        download_all_tags(
            series_ids=series_ids,
            api_key=api_key,
            output_file=tags_dir / "fred_tags_output.parquet",
            checkpoint_file=tags_dir / "fred_tags_checkpoint.json",
            notes_file=tags_dir / "tags_notes_lookup.json",
            max_requests_per_minute=config["tags"]["max_requests_per_minute"],
            concurrent_requests=config["tags"]["concurrent_requests"],
            max_retries=config["tags"]["max_retries"],
            batch_size=config["tags"]["batch_size"],
        )
    )


if __name__ == "__main__":
    main()
