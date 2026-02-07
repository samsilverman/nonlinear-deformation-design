from __future__ import annotations
from typing import Optional, Union, TYPE_CHECKING, Callable, Tuple
import time
from pathlib import Path
import torch
import numpy as np
from matplotlib import pyplot as plt
from utils import load_checkpoint, save_checkpoint, get_device, save_state_dict

if TYPE_CHECKING:
    from os import PathLike
    from torch.nn import Module
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader
    from torch import Tensor


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
                 save_file: Union[str, bytes, PathLike],
                 resume: bool = False,
                 data_transform: Optional[Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]] = None) -> None:
        """Initialize Trainer.

        Parameters
        ----------
        model : torch.nn.Module
            Model.
        criterion : torch.nn.Module
            Criterion.
        optimizer : torch.optim.Optimizer
            Optimizer.
        train_loader : torch.util.data.DataLoader
            Training data loader.
        valid_loader : torch.util.data.DataLoader
            Validation data loader.
        epochs : int
            Number of training epochs.
        save_file : {str, bytes, PathLike}
            Trained model save file (recommended: .pt).
        resume : bool (default=`False`)
            Set to `True` to resume training. To resume training, ``save_file`` must not be `None`.
        data_transform : Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], optional (default=`None`)
            Data augmentation transformation function. It is applied to the ``train_loader`` only.

        """
        self.device_ = get_device()

        self.model_ = model
        self.model_.to(device=self.device_)

        self.criterion_ = criterion
        self.optimizer_ = optimizer
        self.train_loader_ = train_loader
        self.valid_loader_ = valid_loader
        self.epochs_ = epochs
        self.epoch_ = 0
        self.save_file_ = Path(save_file).resolve()
        self.checkpoint_file_ = self.save_file_.with_stem(f'{self.save_file_.stem}-checkpoint')
        self.resume_ = resume
        self.data_transform_ = data_transform
        self.train_losses_ = []
        self.valid_losses_ = []

        if self.resume_ and self.checkpoint_file_.is_file():
            checkpoint_data = load_checkpoint(file=self.checkpoint_file_,
                                              model=self.model_,
                                              optimizer=self.optimizer_)
            self.epoch_, self.train_losses_, self.valid_losses_ = checkpoint_data

            if self.epoch_ >= self.epochs_:
                raise ValueError(f'checkpoint epoch ({self.epoch_}) is greater then epochs ({self.epochs_}).')

    def train(self,
              verbose: bool = True) -> None:
        """Training logic.

        Parameters
        ----------
        verbose : bool (default=`True`)
            Set to `True` to see status messages during training.

        """
        if verbose:
            print(f'{"-" * 5}Training Start (Device: {self.device_}){"-" * 5}')

            training_start = time.time()

        best_loss = float('inf')
        best_epoch = 0

        if len(self.valid_losses_) > 0:
            best_epoch = np.argmin(self.valid_losses_) + 1
            best_loss = self.valid_losses_[best_epoch - 1]

        while self.epoch_ < self.epochs_:
            self.epoch_ += 1

            if verbose:
                print(f'\nEpoch {self.epoch_}/{self.epochs_}')
                print('-' * 10)
                epoch_start = time.time()

            train_loss = self.run_single_epoch_(data_loader=self.train_loader_,
                                                grad_enabled=True)
            valid_loss = self.run_single_epoch_(data_loader=self.valid_loader_,
                                                grad_enabled=False)

            self.train_losses_.append(train_loss)
            self.valid_losses_.append(valid_loss)

            # Early stopping
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = self.epoch_

                save_checkpoint(model=self.model_,
                                optimizer=self.optimizer_,
                                epoch=self.epoch_,
                                train_losses=self.train_losses_,
                                valid_losses=self.valid_losses_,
                                file=self.checkpoint_file_)

            if verbose:
                epoch_elapsed = time.time() - epoch_start
                minutes = int(epoch_elapsed // 60)
                seconds = int(epoch_elapsed % 60)
                milliseconds = int(epoch_elapsed % 1 * 1000)

                print(f'Time: {minutes:02}:{seconds:02}.{milliseconds:03}')
                print(f'Train loss: {train_loss}')
                print(f'Valid loss: {valid_loss}')
                print(f'Best vaid loss: {best_loss}')
                print(f'Best vaid epoch: {best_epoch}')

        # load best model at end of training
        load_checkpoint(file=self.checkpoint_file_,
                        model=self.model_,
                        optimizer=self.optimizer_)

        save_state_dict(model=self.model_, file=self.save_file_)

        if verbose:
            training_elapsed = time.time() - training_start
            hours = int(training_elapsed // 3600)
            minutes = int(training_elapsed % 3600 // 60)
            seconds = int(training_elapsed % 60)
            milliseconds = int(training_elapsed % 1 * 1000)

            print(f'\n{"-" * 5}Training End{"-" * 5}')
            print(f'Time: {hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}')
            print(f'Best valid loss: {best_loss}')
            print(f'Best vaid epoch: {best_epoch}')

            self.visualize_losses_()

    def run_single_epoch_(self,
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
            self.model_.train()
        else:
            self.model_.eval()

        running_loss = 0
        for batch in data_loader:
            inputs, targets = batch

            # Apply data transformation (if applicable) to inputs only
            if self.data_transform_ is not None and grad_enabled:
                inputs, targets = self.data_transform_(inputs, targets)

            inputs = inputs.to(self.device_, non_blocking=True)
            targets = targets.to(self.device_, non_blocking=True)

            if grad_enabled:
                self.optimizer_.zero_grad()

            with torch.set_grad_enabled(mode=grad_enabled):
                outputs = self.model_(inputs)

            loss = self.criterion_(outputs, targets)

            if grad_enabled:
                loss.backward()
                self.optimizer_.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)

        return epoch_loss

    def visualize_losses_(self) -> None:
        """Visualize losses from training.

        """
        figure, axis = plt.subplots()

        epochs = range(self.epochs_)

        axis.plot(epochs, self.train_losses_, label='Training loss')
        axis.plot(epochs, self.valid_losses_, label='Validation loss')

        axis.set_title('Loss Curves')
        axis.set_xlabel('Epoch')
        axis.set_ylabel('Value')
        axis.legend(loc='best')

        figure.tight_layout()
        plt.show()
