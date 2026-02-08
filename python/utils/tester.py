from __future__ import annotations
from typing import TYPE_CHECKING
import time
import torch
from utils import get_device

if TYPE_CHECKING:
    from torch.nn import Module
    from torch.utils.data import DataLoader


class Tester:
    """Testing logic.

    """

    def __init__(self,
                 model: Module,
                 criterion: Module,
                 test_loader: DataLoader) -> None:
        """Initialize Tester.

        Parameters
        ----------
        model : torch.nn.Module
            Model.
        criterion : torch.nn.Module
            Criterion.
        test_loader : torch.utils.data.DataLoader
            Testing DataLoader.

        """
        self.device_ = get_device()

        self.model_ = model
        self.model_.to(device=self.device_)

        self.criterion_ = criterion
        self.test_loader_ = test_loader

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
            Testing loss.

        """
        if verbose:
            print(f'{"-" * 5}Testing start (device: {self.device_}){"-" * 5}')

            start_time = time.time()

        self.model_.eval()

        running_loss = 0

        for batch in self.test_loader_:
            inputs, targets = batch

            inputs = inputs.to(device=self.device_, non_blocking=True)

            if isinstance(targets, list):
                targets = [t.to(device=self.device_, non_blocking=True) for t in targets]
            else:
                targets = targets.to(device=self.device_, non_blocking=True)

            with torch.set_grad_enabled(mode=False):
                outputs = self.model_(inputs)

            loss = self.criterion_(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(self.test_loader_.dataset)

        if verbose:
            elapsed_time = time.time() - start_time
            hours = int(elapsed_time // 3600)
            minutes = int(elapsed_time % 3600 // 60)
            seconds = int(elapsed_time % 60)
            milliseconds = int(elapsed_time % 1 * 1000)

            print(f'{"-" * 5}Testing end{"-" * 5}')
            print(f'Test loss: {epoch_loss}')
            print(f'Time: {hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}')

        return epoch_loss
