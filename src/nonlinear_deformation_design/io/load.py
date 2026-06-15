from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from nonlinear_deformation_design.common import DEFAULT_DEVICE
import torch


if TYPE_CHECKING:
    from collections import OrderedDict
    from pathlib import Path


def load_checkpoint(file: Path, device: torch.device = DEFAULT_DEVICE) -> Tuple[OrderedDict, OrderedDict, OrderedDict, List[float], List[float]]:
    """Loads a model training checkpoint.

    Parameters
    ----------
    file : pathlib.Path
        File. Must have `.pt` extension.
    device : torch.device (default=`DEFAULT_DEVICE`)
        Device.

    Returns
    -------
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

    """
    checkpoint = torch.load(f=file, map_location=device)

    model_state_dict = checkpoint['model_state_dict']
    best_model_state_dict = checkpoint['best_model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    train_losses = checkpoint['train_losses']
    valid_losses = checkpoint['valid_losses']

    return model_state_dict, best_model_state_dict, optimizer_state_dict, train_losses, valid_losses
