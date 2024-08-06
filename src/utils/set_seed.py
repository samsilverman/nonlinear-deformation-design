from __future__ import annotations

import random
import torch
import numpy as np


def set_seed(seed: int) -> None:
    """Sets the seed for generating random numbers.

    Parameters
    ----------
    seed : int
        The seed number.

    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
