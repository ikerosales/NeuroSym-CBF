"""Configuration loading: environment variables (.env) + configs/*.yaml.

All data/model/result paths are resolved from here -- no pipeline module has hardcoded paths,
by design (so it does not depend on Kaggle/Colab).
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


def fred_api_key() -> str:
    """FRED API key.

    The canonical name is `FRED_API_KEY` (the one documented in `.env.example`), but `API_KEY`
    is also accepted as an alias since that's what this project's older `.env` files used.
    """
    for name in ("FRED_API_KEY", "API_KEY"):
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError("Missing FRED_API_KEY in your .env (see .env.example)")


def data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", "./data")).resolve()


def models_dir() -> Path:
    return Path(os.getenv("MODELS_DIR", "./models")).resolve()


def results_dir() -> Path:
    return Path(os.getenv("RESULTS_DIR", "./results")).resolve()


def load_yaml_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
