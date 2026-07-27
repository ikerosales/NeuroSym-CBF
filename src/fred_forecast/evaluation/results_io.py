"""Guarda/carga resultados de evaluación en un formato consistente en todo el proyecto:
métricas agregadas (escalares) en JSON, arrays por muestra en `.npz`.

Evita el patrón del código original de volcar arrays de cientos de miles de
elementos como listas dentro del JSON (`preds_per_sample`, `contexts_per_sample`, ...):
JSON queda pequeño, legible y diffable en git; los arrays van en un binario comprimido,
mucho más rápido de cargar para los scripts de agregación/comparación (`11_aggregate_results.py`).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_SCALAR_TYPES = (int, float, str, bool, type(None))
_JSON_NATIVE_TYPES = (list, dict)


def save_results(
    results: dict,
    json_path: Path,
    npz_path: Path,
    extra_metadata: dict | None = None,
) -> None:
    """Separa `results` en campos escalares (-> `json_path`) y campos array/tensor
    (-> `npz_path`, comprimido). `extra_metadata` se añade siempre al JSON (p.ej. qué
    modelo/contexto/seed generó el resultado).
    """
    scalars: dict = {}
    arrays: dict = {}

    for key, value in results.items():
        if isinstance(value, _SCALAR_TYPES) or isinstance(value, _JSON_NATIVE_TYPES):
            # listas/dicts pequeños (p.ej. IDs de series con fallback) van directos al JSON;
            # para arrays numéricos grandes, usa np.ndarray/torch.Tensor (-> .npz).
            scalars[key] = value
        elif isinstance(value, torch.Tensor):
            arrays[key] = value.numpy()
        elif isinstance(value, np.ndarray):
            arrays[key] = value
        else:
            logger.warning("Campo %r de tipo %s ignorado (ni escalar ni array)", key, type(value))

    if extra_metadata:
        scalars.update(extra_metadata)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(scalars, f, indent=2)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)

    logger.info("Métricas agregadas -> %s", json_path)
    logger.info("Arrays por muestra -> %s", npz_path)


def load_results(json_path: Path, npz_path: Path) -> dict:
    """Recompone el dict de resultados original a partir de los dos ficheros."""
    with open(json_path) as f:
        scalars = json.load(f)
    with np.load(npz_path) as npz:
        arrays = {key: npz[key] for key in npz.files}
    return {**scalars, **arrays}
