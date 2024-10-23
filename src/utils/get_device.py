from __future__ import annotations

import torch

def get_device() -> torch.device:
    """Determines the appropriate device for tensor computations.
    
    Returns
    -------
    device : torch.device
        The device ('cpu', 'cuda', or 'mps').

    """
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    return device
