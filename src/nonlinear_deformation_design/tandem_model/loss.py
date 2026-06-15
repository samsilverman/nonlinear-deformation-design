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


class InverseDesignLoss(nn.Module):
    """Inverse design loss.

    """

    def __init__(self, performance_criterion: nn.Module, design_criterion: nn.Module, alpha: float) -> None:
        """Initialize LILoss.

        Parameters
        ----------
        performance_criterion : torch.nn.Module
            Performance loss model.
        design_criterion : torch.nn.Module
            Design loss model.
        alpha : float
            Weight of `Ld` relative to `Lp`.

        """
        super().__init__()

        self.Lp_ = performance_criterion
        self.Ld_ = design_criterion
        self.alpha_ = alpha

    def forward(self, parameters_outputs: Tensor, performance_outputs: Tensor, parameters_targets: Tensor, performance_targets: Tensor) -> Tensor:
        """Forward propagation.

        Parameters
        ----------
        parameters_outputs : (N, 17) torch.Tensor
            Predicted design parameters.
        performance_outputs : (N, 11) torch.Tensor
            Predicted performance vectors.
        parameters_targets : (N, 17) torch.Tensor
            Target design parameters.
        performance_targets : (N, 11) torch.Tensor
            Target performance vectors.

        Returns
        -------
        loss : (1,) torch.Tensor
            Inverse design loss.

        """
        return self.Lp_(performance_outputs, performance_targets) + self.alpha_ * self.Ld_(parameters_outputs, parameters_targets)
