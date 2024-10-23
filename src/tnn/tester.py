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
            The model.
        criterion : torch.nn.Module
            The criterion.
        test_loader : torch.utils.data.DataLoader
            The testing DataLoader.

        """
        self._model = model
        self._criterion = criterion
        self._test_loader = test_loader

        self._device = get_device()
        self._model.to(device=self._device)

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
            print(f'{"-" * 5}Testing Start{"-" * 5}')

            start_time = time.time()

        self._model.eval()

        running_loss = 0

        for batch in self._test_loader:
            inputs, targets = batch

            inputs = inputs.to(device=self._device)
            targets = targets.to(device=self._device)

            with torch.set_grad_enabled(mode=False):
                outputs = self._model(inputs)

            loss = self._criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

        num_samples = len(self._test_loader) * self._test_loader.batch_size
        epoch_loss = running_loss / num_samples

        if verbose:
            time_elapsed = time.time() - start_time

            print(f'test Loss: {epoch_loss}')
            print(f'{"-" * 5}Testing End{"-" * 5}')
            print(f'Time:{time_elapsed // 60}m {time_elapsed % 60}s')

        return epoch_loss
