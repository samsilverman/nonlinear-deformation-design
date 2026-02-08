#!/usr/bin/env python3
"""Script to get the dataset statistics.

"""
from __future__ import annotations
import numpy as np
from ml import load_data, calculate_stiffness, calculate_work


def main():
    _, performance = load_data()

    stiffness = calculate_stiffness(performance=performance)
    work = calculate_work(performance=performance)
    max_displacement = performance[:, 0]

    # Print overall statistics
    print('Stiffness (N/mm)')
    print(f'\tmean: {np.mean(stiffness)}')
    print(f'\trange: {np.max(stiffness) - np.min(stiffness)}')

    print('Work (J)')
    print(f'\tmean: {np.mean(work)}')
    print(f'\trange: {np.max(work) - np.min(work)}')

    print('Max. displacement (mm)')
    print(f'\tmean: {np.mean(max_displacement)}')
    print(f'\trange: {np.max(max_displacement) - np.min(max_displacement)}')


if __name__ == '__main__':
    main()
