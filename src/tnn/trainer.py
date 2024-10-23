from __future__ import annotations

from typing import Optional, Union, TYPE_CHECKING
import time
from pathlib import Path
import torch
from matplotlib import pyplot as plt
from utils import load_checkpoint, save_checkpoint, get_device

if TYPE_CHECKING:
    from os import PathLike
    from torch.nn import Module
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

class Trainer:
    """Training logic.

    """

    def __init__(self,
                 model: Module,
                 criterion: Module,
                 optimizer: Optimizer,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 epochs: int,
                 file: Optional[Union[str, bytes, PathLike]] = None,
                 resume: bool = False,
                 early_stopping: bool = True) -> None:
        """Initialize Trainer.

        Parameters
        ----------
        model : torch.nn.Module
            The model.
        criterion : torch.nn.Module
            The criterion.
        optimizer : torch.optim.Optimizer
            The optimizer.
        train_loader : torch.util.data.DataLoader
            The training data loader.
        valid_loader : torch.util.data.DataLoader
            The validation data loader.
        epochs : int
            The number of training epochs.
        file : {str, bytes, PathLike}, optional (default=`None`)
            The training checkpoint save file (recommended: .pt). If `None`, no saving occurs.
        resume : bool
            Set to `True` to resume training. To resume training, ``file`` must not be `None`.
        early_stopping : bool
            Set to `True` to use early stopping. To use early stopping, ``file`` must not be `None`.

        Raises
        ------
        ValueError
            If ``resume==True`` and ``loaded_epoch<=epochs``.

        """
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._train_loader = train_loader
        self._valid_loader = valid_loader
        self._epochs = epochs
        self._epoch = 0
        self._file = None
        self._resume = resume
        self._early_stopping = early_stopping
        self._train_losses = []
        self._valid_losses = []

        self._device = get_device()
        self._model.to(device=self._device)

        if file is not None:
            self._file = Path(file).resolve()

        if self._resume and self._file is None:
            print('To resume training, file must be set. Setting resume=False.')
            self._resume = False

        if self._early_stopping and self._file is None:
            print('To perform early stopping, file must be set. Setting early_stopping=False.')
            self._early_stopping = False

        if self._resume:
            checkpoint_data = load_checkpoint(file=self._file,
                                              model=self._model,
                                              optimizer=self._optimizer)
            self._epoch, self._train_losses, self._valid_losses = checkpoint_data

            if self._epoch >= self._epochs:
                raise ValueError(f'checkpoint epoch ({self._epoch}) is greater then epochs ({self._epochs}).')

    def train(self,
              verbose: bool = True) -> None:
        """Testing logic.

        Parameters
        ----------
        verbose : bool (default=`True`)
            Set to `True` to see status messages during training.

        Returns
        -------
        loss : float
            The traning loss.

        """
        if not isinstance(verbose, bool):
            raise TypeError('verbose must be a boolean.')

        if verbose:
            print(f'{"-" * 5}Training Start{"-" * 5}')

            start_time = time.time()

        best_loss = float('inf')

        while self._epoch < self._epochs:
            self._epoch += 1

            if verbose:
                print(f'Epoch {self._epoch}/{self._epochs}')
                print('-' * 10)

            train_loss = self._run_single_epoch(data_loader=self._train_loader,
                                                grad_enabled=True)
            valid_loss = self._run_single_epoch(data_loader=self._valid_loader,
                                                grad_enabled=False)

            self._train_losses.append(train_loss)
            self._valid_losses.append(valid_loss)

            if valid_loss < best_loss:
                best_loss = valid_loss
                if self._early_stopping:
                    save_checkpoint(model=self._model,
                                    optimizer=self._optimizer,
                                    epoch=self._epoch,
                                    train_losses=self._train_losses,
                                    valid_losses=self._valid_losses,
                                    file=self._file)

            if verbose:
                print(f'train Loss: {train_loss}')
                print(f'valid Loss: {valid_loss}')

        # load best model at end of training
        if self._early_stopping:
            load_checkpoint(file=self._file,
                            model=self._model,
                            optimizer=self._optimizer)

        if verbose:
            time_elapsed = time.time() - start_time

            print(f'{"-" * 5}Training End{"-" * 5}')
            print(f'Time:{time_elapsed // 60}m {time_elapsed % 60}s')
            print(f'Best valid loss: {best_loss}')

            self._visualize_losses()

    def _run_single_epoch(self,
                          data_loader: DataLoader,
                          grad_enabled: bool) -> float:
        """Training/validation logic for a single epoch.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The DataLoader.
        grad_enabled : bool
            Set to `True` for training. Set to `False` for validation.

        Returns
        -------
        loss : float
            The average loss for the epoch.

        """
        if grad_enabled:
            self._model.train()
        else:
            self._model.eval()

        running_loss = 0
        for batch in data_loader:
            inputs, targets = batch

            inputs = inputs.to(device=self._device)
            targets = targets.to(device=self._device)

            if grad_enabled:
                self._optimizer.zero_grad()

            with torch.set_grad_enabled(mode=grad_enabled):
                outputs = self._model(inputs)

            loss = self._criterion(outputs, targets)

            if grad_enabled:
                loss.backward()
                self._optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        num_samples = len(data_loader) * data_loader.batch_size
        epoch_loss = running_loss / num_samples

        return epoch_loss

    def _visualize_losses(self) -> None:
        """Visualize losses from training.

        """
        figure, axis = plt.subplots()

        epochs = range(self._epochs)

        axis.plot(epochs, self._train_losses, label='Training loss')
        axis.plot(epochs, self._valid_losses, label='Validation loss')

        axis.set_title('Loss Curves')
        axis.set_xlabel('Epoch')
        axis.set_ylabel('Value')
        axis.legend(loc='best')

        figure.tight_layout()
        plt.show()
