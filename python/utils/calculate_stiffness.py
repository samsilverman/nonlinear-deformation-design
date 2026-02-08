from __future__ import annotations

import numpy as np


def calculate_stiffness(displacements: np.ndarray, forces: np.ndarray) -> np.ndarray:
    """Calculates the stiffness for force-displacement curves.

    Parameters
    ----------
    displacements : (`N`, `M`) numpy.ndarray
        The displacements (mm).
        * `N`: Number of samples.
        * `M`: Number of points in force-displacement curves.
    forces : (`N`, `M`) numpy.ndarray
        The forces (N).

    Returns
    -------
    stiffness : (`N`,) numpy.ndarray
        The stiffness (N / mm).

    """
    # only look at beginning 25% of curves
    x = np.array_split(ary=displacements, indices_or_sections=4, axis=1)[0]
    y = np.array_split(ary=forces, indices_or_sections=4, axis=1)[0]

    stiffness = []

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
