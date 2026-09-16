"""Checkpoint loading, including checkpoints produced by the original notebooks.

The thesis notebooks defined the patch encoder *inside* the model class (`input_proj` and
`res_blocks` as direct attributes), while the package factors it out into `PatchEncoder`
(`models/patch_encoder.py`), so it ends up nested under `encoder.`. The weights are the same and
the shapes match; only the prefix of two key groups changes.

This module detects the old layout and translates it, so a `.pth` exported from Kaggle can be
evaluated with the repo's code without touching it.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Attributes that hung directly off the model class in the notebooks and now live inside PatchEncoder.
_ENCODER_PREFIXES = ("input_proj.", "res_blocks.")

_HYPERPARAM_KEYS = (
    "seed", "context_len", "horizon_len", "d_model", "seasonality", "dict_mult", "top_k", "d_meta",
)


def _is_notebook_layout(state_dict: dict) -> bool:
    return any(k.startswith(_ENCODER_PREFIXES) for k in state_dict)


def remap_notebook_state_dict(state_dict: dict) -> dict:
    """Prefixes encoder keys with `encoder.` if the checkpoint comes from a notebook."""
    if not _is_notebook_layout(state_dict):
        return state_dict
    n_encoder = sum(1 for k in state_dict if k.startswith(_ENCODER_PREFIXES))
    remapped = {
        (f"encoder.{k}" if k.startswith(_ENCODER_PREFIXES) else k): v for k, v in state_dict.items()
    }
    logger.info(
        "Checkpoint in notebook format: %d of %d keys relocated under 'encoder.'",
        n_encoder, len(state_dict),
    )
    return remapped


def load_checkpoint(path: str | Path, map_location="cpu") -> tuple[dict, dict]:
    """Returns `(state_dict ready for load_state_dict, saved hyperparameters)`.

    Accepts both the repo's own checkpoints (`model_state_dict` + hyperparameters) and a bare
    `state_dict`. The hyperparameters let the caller verify the checkpoint matches the protocol
    it is about to be evaluated with.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        hyperparams = {k: checkpoint[k] for k in _HYPERPARAM_KEYS if k in checkpoint}
    else:
        state_dict, hyperparams = checkpoint, {}

    return remap_notebook_state_dict(state_dict), hyperparams


def check_hyperparams(hyperparams: dict, expected: dict) -> None:
    """Warns (without aborting) if the checkpoint was trained with values other than expected."""
    for key, value in expected.items():
        stored = hyperparams.get(key)
        if stored is not None and stored != value:
            logger.warning(
                "Checkpoint was trained with %s=%s, but will be evaluated with %s=%s", key, stored, key, value
            )
