from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load the GCS designs and force-displacement data.

    Returns
    -------
    parameters : (12705, 12) numpy.ndarray
        GCS design parameters.
    performance : (12705, 101) numpy.ndarray
        Uniaxial compression data.
        The first entry is the maximum displacement and
        the following 100 entries are the force data.

    """
    curr_dir = Path(__file__).resolve().parent
    data_dir = curr_dir.parent.parent.parent / 'data'

    parameters = pd.read_csv(filepath_or_buffer=data_dir / 'parameters.csv',
                             delimiter=',',
                             header=0)
    displacements = pd.read_csv(filepath_or_buffer=data_dir / 'displacements.csv',
                                delimiter=',',
                                header=0)
    forces = pd.read_csv(filepath_or_buffer=data_dir / 'forces.csv',
                         delimiter=',',
                         header=0)
    
    parameters = parameters.iloc[:, 1:].to_numpy()
    displacements = displacements.iloc[:, 1:].to_numpy()
    forces = forces.iloc[:, 1:].to_numpy()

    return parameters, np.hstack(tup=(displacements, forces))
