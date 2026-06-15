from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Sets the RNG seed.

    Parameters
    ----------
    seed : int
        RNG seed.

    """
    random.seed(a=seed)

    np.random.seed(seed=seed)

    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed=seed)
