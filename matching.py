"""
Helper functions for solving the matching problem.

- If SciPy is available, uses scipy.optimize.linear_sum_assignment (exact, fast, rectangular).
- Otherwise, falls back to a pure-Python exact Hungarian algorithm (pads to square),
  which is slower but correct (OK for moderate n). For large n (thousands), SciPy is strongly recommended.
"""
from __future__ import annotations

from typing import Tuple, List
import numpy as np
from utils import Allocation, Instance

# SciPy-accelerated rectangular assignment --------------------

def rectangular_assignment_exact(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve min-cost assignment for cost shape (n, m) with n ≤ m.
       Returns (rows, cols) indices of matches (size n).
    """
    n, m = cost.shape
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        r, c = linear_sum_assignment(cost)
        return r, c
    except Exception:
        # Fallback: pad to square and run Hungarian (O(N^3) Python)
        return _hungarian_rect_fallback(cost)

# Hungarian fallback ------------------------------------------

def _hungarian_rect_fallback(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n, m = cost.shape
    N = max(n, m)
    pad = np.full((N, N), cost.max() + 1.0, dtype=float)
    pad[:n, :m] = cost
    r, c = _hungarian_square(pad)
    mask = r < n
    return r[mask], c[mask]


def _hungarian_square(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    C = cost.copy()
    N = C.shape[0]
    C -= C.min(axis=1, keepdims=True)
    C -= C.min(axis=0, keepdims=True)

    starred = np.zeros_like(C, dtype=bool)
    primed = np.zeros_like(C, dtype=bool)
    covered_rows = np.zeros(N, dtype=bool)
    covered_cols = np.zeros(N, dtype=bool)

    # initial stars
    for i in range(N):
        for j in range(N):
            if (C[i, j] == 0) and (not covered_rows[i]) and (not covered_cols[j]):
                starred[i, j] = True
                covered_rows[i] = True
                covered_cols[j] = True
    covered_rows[:] = False
    covered_cols[:] = False

    def cover_starred_cols():
        covered_cols[:] = np.any(starred, axis=0)

    cover_starred_cols()

    def find_zero():
        for i in range(N):
            if not covered_rows[i]:
                for j in range(N):
                    if (not covered_cols[j]) and (C[i, j] == 0) and (not starred[i, j]) and (not primed[i, j]):
                        return i, j
        return None

    def find_star_in_row(i):
        cols = np.where(starred[i])[0]
        return (cols[0] if cols.size else None)

    def find_star_in_col(j):
        rows = np.where(starred[:, j])[0]
        return (rows[0] if rows.size else None)

    def find_prime_in_row(i):
        cols = np.where(primed[i])[0]
        return (cols[0] if cols.size else None)

    def augment_path(path):
        for (i, j) in path:
            starred[i, j] = not starred[i, j]
            primed[i, j] = False

    def clear_primes():
        primed[:] = False

    while True:
        if covered_cols.sum() == N:
            break
        z = find_zero()
        while z is None:
            uncovered = (~covered_rows)[:, None] & (~covered_cols)[None, :]
            min_uncovered = np.min(C[uncovered])
            C[~covered_rows, :] -= min_uncovered
            C[:, covered_cols] += min_uncovered
            z = find_zero()
        i, j = z
        primed[i, j] = True
        s = find_star_in_row(i)
        if s is None:
            path = [(i, j)]
            k = j
            r = find_star_in_col(k)
            while r is not None:
                path.append((r, k))
                k2 = find_prime_in_row(r)
                path.append((r, k2))
                k = k2
                r = find_star_in_col(k)
            augment_path(path)
            clear_primes()
            covered_rows[:] = False
            covered_cols[:] = False
            cover_starred_cols()
        else:
            covered_rows[i] = True
            covered_cols[s] = False

    rows = np.arange(N)
    cols = np.argmax(starred, axis=1)
    return rows, cols
