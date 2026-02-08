from __future__ import annotations
import numpy as np


def calculate_stiffness(performance: np.ndarray) -> np.ndarray:
    """Calculates the stiffness for force-displacement curves.

    Parameters
    ----------
    performance : (N, 101) numpy.ndarray
        Uniaxial compression data. Columns:
        - 1: Maximum displacement
        - 2...101: Forces

    Returns
    -------
    stiffnesses : (N,) numpy.ndarray
        Stiffness (N/mm) for force-displacement curves.

    """
    max_displacements = performance[:, 0].reshape(-1, 1)
    displacements = max_displacements * np.linspace(start=0, stop=1, num=100)
    forces = performance[:, 1:]

    # only look at beginning 25% of curves
    x = np.array_split(ary=displacements, indices_or_sections=4, axis=1)[0]
    y = np.array_split(ary=forces, indices_or_sections=4, axis=1)[0]

    stiffness = []

    # Report the stiffness as the maximum slope for data windows of size 5
    window_size = 5

    for index in range(x.shape[0]):
        slopes = []
        for start in range(x.shape[1] - window_size):
            stop = start + window_size

            x_window = x[index, start:stop]
            y_window = y[index, start:stop]

            fit = np.polyfit(x=x_window, y=y_window, deg=1)

            slopes.append(fit[0])

        stiffness.append(np.max(slopes))

    stiffness = np.array(stiffness)

    return stiffness
