from __future__ import annotations
from typing import Union, List, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from os import PathLike
    from torch.nn import Module
    from torch.optim import Optimizer


def save_checkpoint(model: Module,
                    optimizer: Optimizer,
                    epoch: int,
                    train_losses: List[float],
                    valid_losses: List[float],
                    file: Union[str, bytes, PathLike]) -> None:
    """Saves a model training checkpoint.

    Parameters
    ----------
    model : torch.nn.Module
        Model.
    optimizer : torch.optim.Optimizer
        Optimizer.
    epoch : int
        Training epoch.
    train_losses : List[float]
        Training losses at each epoch.
    valid_losses : List[float]
        Validation losses at each epoch.
    file : {str, bytes, PathLike}
        File (recommended: `.pt`).

    """

    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()

    checkpoint = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'epoch': epoch,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
    }

    torch.save(obj=checkpoint, f=file)


def save_state_dict(model: Module,
                    file: Union[str, bytes, PathLike]) -> None:
    """Saves a ``torch.nn.Module`` state dictionary.

    Parameters
    ----------
    model : torch.nn.Module
        Model.
    file : {str, bytes, PathLike}
        File (recommended: `.pt`).

    """
    state_dict = model.state_dict()
    torch.save(obj=state_dict, f=file)
