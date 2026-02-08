from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor


class WeightedMSELoss(nn.Module):
    """Weighted mean squared error (MSE) loss.

    """

    def __init__(self, weights: Tensor) -> None:
        """Initialize WeightedMSELoss.

        Parameters
        ----------
        weights : (1, M) torch.Tensor
            Weights.

        """
        super().__init__()

        if weights.ndim == 1:
            weights = weights.reshape(shape=(1, -1))

        self.weights_ = weights

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
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
            Weighted MSE loss.

        """
        if self.weights_.device != outputs.device:
            self.weights_ = self.weights_.to(device=outputs.device)
        if self.weights_.dtype != outputs.dtype:
            self.weights_ = self.weights_.to(dtype=outputs.dtype)

        return (self.weights_ * (outputs - targets) ** 2).mean()


class LILoss(nn.Module):
    """Inverse design loss.

    """

    def __init__(self, Lp: nn.Module, Ld: nn.Module, alpha: float) -> None:
        """Initialize LILoss.

        Parameters
        ----------
        Lp : torch.nn.Module
            Performance loss model.
        Ld : torch.nn.Module
            Design loss model.
        alpha : float
            Weight of `Ld` relative to `Lp`.

        """
        super().__init__()

        self.Lp_ = Lp
        self.Ld_ = Ld
        self.alpha_ = alpha

    def forward(self, outputs: Tuple[Tensor, Tensor], targets: Tuple[Tensor, Tensor]) -> Tensor:
        """Forward propagation.

        Parameters
        ----------
        outputs : Tuple[torch.Tensor, torch.Tensor]
            Model outputs as a tuple:
            - First item is (N, 17) predicted design vectors.
            - Second item is (N, 11) predicted performance vectors.
        targets : Tuple[torch.Tensor, torch.Tensor]
            Model targets as a tuple:
            - First item is (N, 17) target design vectors.
            - Second item is (N, 11) target performance vectors.

        Returns
        -------
        loss : (1,) torch.Tensor
            Inverse design loss.

        """
        return self.Ld_(outputs[1], targets[1]) + self.alpha_ * self.Ld_(outputs[0], targets[0])
