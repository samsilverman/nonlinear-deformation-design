from __future__ import annotations

from pathlib import Path
from typing import List, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections import OrderedDict


def save_checkpoint(model_state_dict: OrderedDict,
                    best_model_state_dict: OrderedDict,
                    optimizer_state_dict: OrderedDict,
                    train_losses: List[float],
                    valid_losses: List[float],
                    file: Path) -> None:
    """Saves a model training checkpoint.

    Parameters
    ----------
    model_state_dict : collections.OrderedDict
        Model weights.
    best_model_state_dict : collections.OrderedDict
        Best epoch model weights.
    optimizer_state_dict : collections.OrderedDict
        Optimizer weights.
    train_losses : List[float]
        Training losses at each epoch.
    valid_losses : List[float]
        Validation losses at each epoch.
    file : pathlib.Path
        File. Must have `.pt` extension.

    """
    checkpoint = {
        'model_state_dict': model_state_dict,
        'best_model_state_dict': best_model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
    }

    torch.save(obj=checkpoint, f=file)
