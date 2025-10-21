from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class Allocation:
    bundles: List[List[int]]
    
    def to_matrix(self, n: int, m: int) -> np.ndarray:
        X = np.zeros((n, m), dtype=int)
        for i, items in enumerate(self.bundles):
            if items:
                X[i, items] = 1
        return X

@dataclass
class Instance:
    n: int
    m: int
    U: np.ndarray   # shape (n,m)
    kind: str       # 'goods' or 'chores'
