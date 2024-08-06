from __future__ import annotations

from typing import Tuple
import numpy as np


def split_indices(num_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits indices into training, validation, and testing sets.

    Parameters
    ----------
    num_samples : int
        The number of samples.

    Returns
    -------
    train_indices : numpy.ndarray
        The training data indices.
    valid_indices : numpy.ndarray
        The validation data indices.
    test_indices : numpy.ndarray
        The testing data indices.

    """
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    n_train = int(num_samples * 0.8)
    n_valid = int(num_samples * 0.1)

    train_indices = indices[:n_train]
    valid_indices = indices[n_train:n_train+n_valid]
    test_indices = indices[n_train+n_valid:]

    return train_indices, valid_indices, test_indices
