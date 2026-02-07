from __future__ import annotations
from typing import Tuple
import numpy as np


def split_indices(num_samples: int, percent_train: float = 0.8, percent_valid: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits indices into training, validation, and testing sets.

    Parameters
    ----------
    num_samples : int
        Number of samples.
    percent_train : float (default=`0.8`)
        Percentage of training data.
        Combines with `percent_valid` to infer percentage of testing data.
    percent_valid : float (default=`0.1`)
        Percentage of validaion data.
        Combines with `percent_train` to infer percentage of testing data.

    Returns
    -------
    train_indices : numpy.ndarray
        Training indices.
    valid_indices : numpy.ndarray
        Validation indices.
    test_indices : numpy.ndarray
        Testing data indices.

    """
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    n_train = int(num_samples * percent_train)
    n_valid = int(num_samples * percent_valid)

    train_indices = indices[:n_train]
    valid_indices = indices[n_train:n_train+n_valid]
    test_indices = indices[n_train+n_valid:]

    return train_indices, valid_indices, test_indices
