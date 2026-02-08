from __future__ import annotations
from typing import List
from pathlib import Path


def load_seeds() -> List[int]:
    """Load random seeds from the `models/seeds.txt` file.

    Returns
    -------
    seeds : List[int]
        Seeds.

    """
    file = Path(__file__).resolve().parent.parent.parent / 'models' / 'seeds.txt'

    seeds = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            seeds.append(int(line.strip()))

    return seeds
