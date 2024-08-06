from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.nn import Module


def print_model(model: Module) -> None:
    """Print a model.

    Parameters
    ----------
    model : torch.nn.Module
        The model.

    """
    total_parameters = 0
    trainable_parameters = 0

    for layer in model.parameters():
        num_parameters = layer.numel()
        total_parameters += num_parameters

        if layer.requires_grad:
            trainable_parameters += num_parameters

    horizontal_bar = '=' * 20

    print(
        f'{horizontal_bar}\n'
        + str(model)
        + f'\n{horizontal_bar}'
        + f'\nTotal parameters: {total_parameters}'
        + f'\nTrainable parameters: {trainable_parameters}'
        + f'\nNon-trainable parameters: {total_parameters - trainable_parameters}'
        + f'\n{horizontal_bar}'
    )
