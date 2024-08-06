from __future__ import annotations

from typing import Union, TYPE_CHECKING, Tuple
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    import numpy as np


class WeightedMSELoss(nn.Module):
    """Weighted mean squared error (MSE) loss.

    """

    def __init__(self,
                 weights: Union[Tensor, np.ndarray]) -> None:
        """Initialize WeightedMSELoss.

        Parameters
        ----------
        weights : (1, M) {torch.Tensor, numpy.ndarray}
            The weights.

        """
        super().__init__()

        if not torch.is_tensor(weights):
            weights = torch.from_numpy(weights)

        if weights.ndim == 1:
            weights = weights.reshape(shape=(1, -1))

        self._weights = weights

    def forward(self,
                outputs: Tensor,
                targets: Tensor) -> float:
        """Forward propagation.

        Parameters
        ----------
        outputs : (N, M) torch.Tensor
            Model outputs.
        targets : (N, M) torch.Tensor
            Model targets.

        Returns
        -------
        loss : (1,) torch.Tensor
            The loss.

        """
        loss = (self._weights * (outputs - targets) ** 2).mean()

        return loss


class LILoss(nn.Module):
    """Inverse design loss.

    """

    def __init__(self,
                 Lp: nn.Module,
                 Ld: nn.Module,
                 alpha: float) -> None:
        """Initialize LILoss.

        Parameters
        ----------
        Lp : torch.nn.Module
            The performance loss model.
        Ld : torch.nn.Module
            The deisgn loss model.
        alpha : float
            The alpha value.

        """
        super().__init__()

        self._Lp = Lp
        self._Ld = Ld
        self._alpha = alpha

    def forward(self,
                outputs: Tuple[Tensor, Tensor],
                targets: Tuple[Tensor, Tensor]) -> float:
        """Forward propagation.

        Parameters
        ----------
        outputs : Tuple[torch.Tensor, torch.Tensor]
            Model outputs.
        targets : (N, M) torch.Tensor
            Model targets.

        Returns
        -------
        loss : (1,) torch.Tensor
            The loss.

        """
        loss = self._Lp(outputs[1], targets[1]) + self._alpha * self._Ld(outputs[0], targets[0])

        return loss
