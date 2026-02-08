from __future__ import annotations
from typing import Tuple
from pathlib import Path
import numpy as np
import pandas as pd

def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load the data.

    Returns
    -------
    parameters : (N, 16) numpy.ndarray
        GCS design parameters.
    performance : (N, 101) numpy.ndarray
        Uniaxial compression data. Columns:
        - 1: Maximum displacement
        - 2...101: Forces

    """
    data_folder = Path(__file__).parent.parent.parent.resolve() / 'data'

    parameters = pd.read_csv(filepath_or_buffer=data_folder / 'parameters.csv',
                             delimiter=',',
                             header=0)

    displacements = pd.read_csv(filepath_or_buffer=data_folder / 'displacements.csv',
                                delimiter=',',
                                header=0)

    forces = pd.read_csv(filepath_or_buffer=data_folder / 'forces.csv',
                         delimiter=',',
                         header=0)

    parameters = parameters.iloc[:, 1:].to_numpy()
    displacements = displacements.iloc[:, 1:].to_numpy()
    forces = forces.iloc[:, 1:].to_numpy()

    return parameters, np.hstack(tup=(displacements, forces))
