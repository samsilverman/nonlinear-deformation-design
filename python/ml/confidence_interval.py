from __future__ import annotations
from typing import List, Tuple
import scipy.stats as stats
import numpy as np


def confidence_interval(data: List[float]) -> Tuple[float, float]:
    """Calculates the mean and 95% confidence interval margin for a list of data.

    Parameters
    ----------
    data : List[float]
        Data.

    Returns
    -------
    mu : float
        Mean value of `data`.
    margin : float
        95% confidence interval margin for `data`.

    """
    mean = np.mean(data)

    interval = stats.t.interval(confidence=0.95,
                                df=len(data) - 1,
                                loc=mean,
                                scale=stats.sem(data))

    margin = (interval[1] - interval[0]) / 2

    return mean, margin
