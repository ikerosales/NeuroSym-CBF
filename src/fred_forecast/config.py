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


def get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing environment variable {name!r} (add it to your .env, see the README)")
    return value


def data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", "./data")).resolve()


def models_dir() -> Path:
    return Path(os.getenv("MODELS_DIR", "./models")).resolve()


def results_dir() -> Path:
    return Path(os.getenv("RESULTS_DIR", "./results")).resolve()


def load_yaml_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
