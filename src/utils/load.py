from __future__ import annotations

from typing import Union, TYPE_CHECKING, List, Tuple
from pathlib import Path
import pandas as pd
import torch

if TYPE_CHECKING:
    from os import PathLike
    import numpy as np
    from torch.nn import Module
    from torch.optim import Optimizer


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the data.

    Returns
    -------
    parameters : (N, 16) numpy.ndarray
        The parameters data.
    displacements : (N, 1) numpy.ndarray
        The displacements data.
    forces : (N, 100) numpy.ndarray
        The forces data.

    """
    data_folder = Path(__file__).parent.parent.parent.resolve() / 'data'

    parameters = pd.read_csv(filepath_or_buffer=data_folder / 'parameters.csv',
                             delimiter=',',
                             header=0)

    displacements = pd.read_csv(filepath_or_buffer=data_folder / 'displacements.csv',
                                delimiter=',',
                                header=0)

    forces = pd.read_csv(filepath_or_buffer=data_folder / 'forces.csv',
                         delimiter=',',
                         header=0)

    parameters = parameters.iloc[:, 1:].to_numpy()
    displacements = displacements.iloc[:, 1:].to_numpy()
    forces = forces.iloc[:, 1:].to_numpy()

    return parameters, displacements, forces


def load_checkpoint(file: Union[str, bytes, PathLike],
                    model: Module,
                    optimizer: Optimizer) -> Tuple[int, List[int], List[int]]:
    """Loads a model training checkpoint.

    Parameters
    ----------
    file : {str, bytes, PathLike}
        The file (recommended: `.pt`).

    Returns
    -------
    epoch : int
        The checkpoint epoch.
    train_losses : List[int]
        The training losses of each epoch.
    valid_losses : List[int]
        The validation losses of each epoch.

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
        The model.
    file : {str, bytes, PathLike}
        The file (recommended: `.pt`).

    """
    state_dict = torch.load(f=file)
    model.load_state_dict(state_dict=state_dict)
