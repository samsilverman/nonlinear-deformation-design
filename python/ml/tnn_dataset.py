from __future__ import annotations
from typing import Tuple
import torch
from torch.utils.data import Dataset


class TNNDataset(Dataset):
    """Custom `torch.utils.data.Dataset` for inverse design training."""

    def __init__(self, performance: torch.Tensor, parameters: torch.Tensor) -> None:
        """Initialize TNNDataset.

        Parameters
        ----------
        performance : (`N`, 17) torch.Tensor
            The performance vectors.
            * `N`: Number of samples.
        parameters : (`N`, 11) torch.Tensor
            The parameter vectors.

        """
        self._performance = performance
        self._parameters = parameters

    def __len__(self) -> int:
        return self._parameters.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        performance = self._performance[index]
        parameters = self._parameters[index]

        return performance, (parameters, performance)
