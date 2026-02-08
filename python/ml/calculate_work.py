from __future__ import annotations
import numpy as np


def calculate_work(performance: np.ndarray) -> np.ndarray:
    """Calculates the work for force-displacement curves.

    Parameters
    ----------
    performance : (N, 101) numpy.ndarray
        Uniaxial compression data. Columns:
        - 1: Maximum displacement
        - 2...101: Forces

    Returns
    -------
    work : (N,) numpy.ndarray
        Work (J) for force-displacement curves.

    """
    max_displacements = performance[:, 0].reshape(-1, 1)
    displacements = max_displacements * np.linspace(start=0, stop=1, num=100)
    forces = performance[:, 1:]

    work = np.trapz(y=forces, x=displacements, axis=1)

    # mJ to J
    work /= 1000

    return work
