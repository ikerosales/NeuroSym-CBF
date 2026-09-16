"""Environment pinning so results are reproducible.

Replicates the header cell of the original notebooks ("Reproducibility -- FIXED across all
notebooks for fair comparison"), in a single place instead of copied into every script.

The CUDA/cuDNN calls are no-ops on a PyTorch install without a GPU (which is where the thesis was
trained, both on Kaggle and locally), but they are kept so the repo stays reproducible if someone
runs it on a GPU.

`num_threads` is not a performance parameter: the number of threads changes the reduction order
of matmuls on CPU and, with it, the training result. Measured on this machine, with the same
seed and the same data, the batch-501 loss goes from 0.3361 (4 threads) to 0.3373 (1 thread); two
runs with the same value match exactly. Evaluation is insensitive to this (MAE identical to 8
decimals), but it is pinned anyway for consistency.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_reproducible_environment(seed: int, num_threads: int | None = None) -> torch.device:
    """Seeds the RNGs, pins cuDNN determinism and the number of CPU threads.

    Returns the device to use, so the caller does not have to repeat the check. Must be called
    before building the model: weight initialization consumes the global RNG.
    """
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op without CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | threads: %d | seed: %d", device, torch.get_num_threads(), seed)
    return device
