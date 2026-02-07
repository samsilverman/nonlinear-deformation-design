from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Callable, Tuple
import time
import torch
from utils import get_device

if TYPE_CHECKING:
    from torch.nn import Module
    from torch import Tensor
    from torch.utils.data import DataLoader


class Tester:
    """Testing logic.

    """

    def __init__(self,
                 model: Module,
                 criterion: Module,
                 test_loader: DataLoader,
                 data_transform: Optional[Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]] = None) -> None:
        """Initialize Tester.

        Parameters
        ----------
        model : torch.nn.Module
            Model.
        criterion : torch.nn.Module
            Criterion.
        test_loader : torch.utils.data.DataLoader
            Testing DataLoader.
        data_transform : Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]], optional (default=`None`)
            Data augmentation transformation function.

        """
        self.device_ = get_device()

        self.model_ = model
        self.model_.to(device=self.device_)

        self.criterion_ = criterion
        self.test_loader_ = test_loader
        self.data_transform_ = data_transform

    def test(self,
             verbose: bool = True) -> float:
        """Testing logic.

        Parameters
        ----------
        verbose : bool (default=`True`)
            Set to `True` to see status messages during testing.

        Returns
        -------
        loss : float
            The testing loss.

        """
        if verbose:
            print(f'{"-" * 5}Testing Start (Device: {self.device_}){"-" * 5}')

            testing_start = time.time()

        self.model_.eval()

        running_loss = 0

        for batch in self.test_loader_:
            inputs, targets = batch

            if self.data_transform_ is not None:
                inputs, targets = self.data_transform_(inputs, targets)

            inputs = inputs.to(self.device_, non_blocking=True)
            targets = targets.to(self.device_, non_blocking=True)

            with torch.set_grad_enabled(mode=False):
                outputs = self.model_(inputs)

            loss = self.criterion_(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(self.test_loader_.dataset)

        if verbose:
            training_elapsed = time.time() - testing_start
            minutes = int(training_elapsed // 60)
            seconds = int(training_elapsed % 60)
            milliseconds = int(training_elapsed % 1 * 1000)

            print(f'test loss: {epoch_loss}')
            print(f'{"-" * 5}Testing End{"-" * 5}')
            print(f'Time: {minutes:02}:{seconds:02}.{milliseconds:03}')

        return epoch_loss
