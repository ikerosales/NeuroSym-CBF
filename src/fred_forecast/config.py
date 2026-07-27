"""Carga de configuración: variables de entorno (.env) + configs/*.yaml.

Todas las rutas de datos/modelos/resultados se resuelven desde aquí — ningún módulo del
pipeline tiene rutas hardcodeadas, por diseño (para no depender de Kaggle/Colab).
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
        raise RuntimeError(f"Falta la variable de entorno {name!r} (créala en tu .env, ver el README)")
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
