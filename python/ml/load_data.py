from __future__ import annotations
from typing import Tuple
from pathlib import Path
import pandas as pd


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the data.

    Returns
    -------
    parameters : (12705, 12) numpy.ndarray
        GCS design parameters.
    performance : (12705, 101) numpy.ndarray
        GCS uniaxial compression data. Columns:
        - 1: Maximum displacement
        - 2...101: Forces

    """
    data_dir = Path(__file__).resolve().parent.parent.parent / 'data'

    parameters = pd.read_csv(filepath_or_buffer=data_dir / 'parameters.csv',
                             delimiter=',',
                             header=0)

    max_displacements = pd.read_csv(filepath_or_buffer=data_dir / 'displacements.csv',
                                delimiter=',',
                                index_col=None,
                                header=0)

    forces = pd.read_csv(filepath_or_buffer=data_dir / 'forces.csv',
                         delimiter=',',
                         header=0)

    # Remove ID_Number column
    parameters = parameters.iloc[:, 1:]
    max_displacements = max_displacements.iloc[:, 1:]
    forces = forces.iloc[:, 1:]

    return parameters, pd.concat(objs=(max_displacements, forces), axis=1)
