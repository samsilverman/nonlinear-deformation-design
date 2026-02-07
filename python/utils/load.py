from __future__ import annotations
from typing import Union, TYPE_CHECKING, List, Tuple
import torch
from utils import get_device


if TYPE_CHECKING:
    from os import PathLike
    from torch.nn import Module
    from torch.optim import Optimizer


def load_checkpoint(file: Union[str, bytes, PathLike],
                    model: Module,
                    optimizer: Optimizer) -> Tuple[int, List[int], List[int]]:
    """Loads a model training checkpoint.

    Parameters
    ----------
    file : {str, bytes, PathLike}
        File (recommended: `.pt`).
    model : torch.nn.Module
        Model.
    optimizer : torch.optim.Optimizer
        Optimizer.

    Returns
    -------
    epoch : int
        Checkpoint epoch.
    train_losses : List[int]
        Training losses of each epoch.
    valid_losses : List[int]
        Validation losses of each epoch.

    """
    checkpoint = torch.load(f=file)

    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    valid_losses = checkpoint['valid_losses']

    model.load_state_dict(state_dict=model_state_dict)
    optimizer.load_state_dict(state_dict=optimizer_state_dict)

    return epoch, train_losses, valid_losses


def load_state_dict(model: Module,
                    file: Union[str, bytes, PathLike]) -> None:
    """Loads a ``torch.nn.Module`` state dictionary.

    Parameters
    ----------
    model : torch.nn.Module
        Model.
    file : {str, bytes, PathLike}
        File (recommended: `.pt`).

    """
    device = get_device()
    state_dict = torch.load(f=file, map_location=device)
    model.load_state_dict(state_dict=state_dict)
    model.to(device=device)
