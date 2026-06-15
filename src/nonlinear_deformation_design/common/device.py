from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Determines the appropriate device for tensor computations.

    Returns
    -------
    device : torch.device
        Torch device.

    """
    if torch.cuda.is_available():
        return torch.device('cuda')

    if torch.backends.mps.is_available():
        return torch.device('mps')

    return torch.device('cpu')
