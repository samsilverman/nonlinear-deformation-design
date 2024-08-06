from __future__ import annotations

import numpy as np


def calculate_work(displacements: np.ndarray, forces: np.ndarray) -> np.ndarray:
    """Calculates the work for force-displacement curves.

    Parameters
    ----------
    displacements : (N, M) np.ndarray
        The displacements (mm).
    forces : (N, M) np.ndarray
        The forces (N).

    Returns
    -------
    work : (N,) np.ndarray
        The work (J).

    """
    work = np.trapz(y=forces, x=displacements, axis=1)
    # mJ to J
    work /= 1000

    return work
