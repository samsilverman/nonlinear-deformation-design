from __future__ import annotations

import numpy as np


def calculate_work(displacements: np.ndarray, forces: np.ndarray) -> np.ndarray:
    """Calculates the work for force-displacement curves.

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
    work : (`N`,) numpy.ndarray
        The work (J).

    """
    work = np.trapz(y=forces, x=displacements, axis=1)

    # mJ to J
    work /= 1000

    return work
