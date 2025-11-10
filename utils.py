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

# Envy metrics -------------------------------------------------

def envy_metrics(inst: Instance, alloc: Allocation):
    U = inst.U
    n, m = U.shape
    X = alloc.to_matrix(n, m)              
    V = U @ X.T                            
    own = np.diag(V)                     

    tol = 1e-12

    if inst.kind == 'goods':
        denom = own[:, None]              
        numer = V                          

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(numer, denom)
            mask_d0 = np.isclose(denom, 0.0)
            ratio = np.where(mask_d0 & (numer > tol), np.inf, ratio)
            ratio = np.where(mask_d0 & np.isclose(numer, 0.0), 1.0, ratio)

        ratio = np.maximum(1.0, ratio)

    else:
        d = -V
        d_own = -own                      
        denom = d                          
        numer = d_own[:, None]             

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(numer, denom)
            mask_d0 = np.isclose(denom, 0.0)
            ratio = np.where(mask_d0 & (numer > tol), np.inf, ratio)
            ratio = np.where(mask_d0 & np.isclose(numer, 0.0), 1.0, ratio)

        ratio = np.maximum(1.0, ratio)

    np.fill_diagonal(ratio, 1.0)

    worst_envy_ratio = float(np.nanmax(ratio))

    agents_with_envy = np.any(ratio > 1.0 + 1e-9, axis=1)
    fraction_with_envy = float(np.mean(agents_with_envy))

    return (worst_envy_ratio, fraction_with_envy)
