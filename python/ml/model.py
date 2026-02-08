from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class TNN(nn.Module):
    """Tandem neural network (TNN) model.

    """

    def __init__(self) -> None:
        """Initialize TNN.

        """
        super().__init__()

        # Forward design network
        self.f_ = nn.Sequential(
            nn.Linear(in_features=17, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=11),
        )

        # Inverse design network
        self.i_ = nn.Sequential(
            nn.Linear(in_features=11, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=17)
        )

    def f(self) -> nn.Sequential:
        """Forward design network.

        """
        return self.f_

    def freeze_f(self,
                 freeze: bool = True) -> None:
        """Freeze the layers of the the forward design model.

        Parameters
        ----------
        freeze : bool (default=`True`)
            Set to `True` to freeze or `False` to unfreeze.

        """
        for parameter in self.f_.parameters():
            parameter.requires_grad = not freeze

    def forward_design(self,
                       inputs: Tensor) -> Tensor:
        """Forward design.

        Parameters
        ----------
        inputs : (N, 17) torch.Tensor
            Design vectors.

        Returns
        -------
        outputs : (N, 11) torch.Tensor
            Predicted performance vectors.

        """
        outputs = self.f_(inputs)
        return outputs

    def inverse_design(self,
                       inputs: Tensor,
                       binarize: bool = False) -> Tensor:
        """Inverse design.

        Parameters
        ----------
        inputs : (N, 11) torch.Tensor
            Performance vectors.
        binarize : bool (sefault=`False`)
            Set to `True` to binarize material predictions.

        Returns
        -------
        outputs : (N, 17) torch.Tensor
            Predicted design vectors.

        """
        outputs = self.i_(inputs)
        outputs[:, :11] = torch.sigmoid(outputs[:, :11])
        outputs[:, 11:] = torch.softmax(outputs[:, 11:], dim=1)

        if binarize:
            indices = torch.argmax(input=outputs[:, 11:], dim=1)
            outputs[:, 11:] = torch.nn.functional.one_hot(indices, num_classes=6)

        return outputs

    def forward(self,
                inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward propagation.

        Parameters
        ----------
        inputs : (N, 11) torch.Tensor
            Performance vectors.

        Returns
        -------
        outputs_d : (N, 17) torch.Tensor
            Predicted design vectors.
        outputs_p : (N, 11) torch.Tensor
            Predicted performance vectors.

        """
        outputs_d = self.inverse_design(inputs)
        outputs_p = self.forward_design(outputs_d)
        return outputs_d, outputs_p
