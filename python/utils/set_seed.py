from __future__ import annotations
import random
import torch
import numpy as np


def set_seed(seed: int) -> None:
    """Sets the seed for generating random numbers.

    Parameters
    ----------
    seed : int
        Seed number.

    """
    torch.manual_seed(seed=seed)
    random.seed(a=seed)
    np.random.seed(seed=seed)
