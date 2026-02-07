from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.nn import Module


def print_model(model: Module) -> None:
    """Print a model.

    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

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
        + f'\nModel device: {device}'
        + f'\nModel dtype: {dtype}'
        + f'\nTotal parameters: {total_parameters}'
        + f'\nTrainable parameters: {trainable_parameters}'
        + f'\nNon-trainable parameters: {total_parameters - trainable_parameters}'
        + f'\n{horizontal_bar}'
    )
