"""
Algorithms required by the paper:
 - Large regime (m = Ω(n log n)): maximum-utility per item
   * goods: assign to argmax per item
   * chores: assign to argmin per item
 - Large regime sampled version: sample O(log n) agents per item, assign to best among them
 - Small regime (m = O(n log n)): matching-based (one-to-one)
   Implemented in matching.py (exact).

Only numpy is required here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

from matching import rectangular_assignment_exact
from utils import Allocation, Instance

# Large-regime algorithms --------------------------------------

def alg_max_per_item(inst: Instance) -> Allocation:
    U = inst.U
    n, m = U.shape
    if inst.kind == 'goods':
        winners = np.argmax(U, axis=0)
    else:  # chores
        winners = np.argmin(U, axis=0)
    bundles = [[] for _ in range(n)]
    for j, i in enumerate(winners):
        bundles[int(i)].append(j)
    return Allocation(bundles)

def alg_max_per_item_sampled(inst: Instance, num_samples: int) -> Allocation:
    U = inst.U
    n, m = U.shape
    num_samples = min(num_samples, n)
    bundles = [[] for _ in range(n)]
    for j in range(m):
        sampled_agents = np.random.choice(n, size=num_samples, replace=False)
        sampled_utilities = U[sampled_agents, j]
        if inst.kind == 'goods':
            winner_idx = np.argmax(sampled_utilities)
        else:  # chores
            winner_idx = np.argmin(sampled_utilities)
        winner = sampled_agents[winner_idx]
        bundles[int(winner)].append(j)
    return Allocation(bundles)

# Small-regime algorithm --------------------------------------

def alg_matching(inst: Instance) -> Allocation:
    """Multi-round one-to-one matchings so each agent gets x or x+1 items.
       If m = x*n + y (0 ≤ y < n), run x full 1-1 matchings, then 1 more on the y leftover items.
       Goods: maximize total utility each round; Chores: minimize total disutility each round.
    """
    U = inst.U
    n, m = U.shape
    if n == 0 or m == 0:
        return Allocation([[] for _ in range(n)])

    x, y = divmod(m, n)

    remaining = list(range(m))
    bundles = [[] for _ in range(n)]

    # Helper to build round cost matrix with current remaining items
    def build_cost(subcols: list[int]) -> np.ndarray:
        M = U[:, subcols]
        if inst.kind == 'goods':
            shift = float(M.max()) if M.size else 0.0
            return (shift - M) 
        else:  # chores
            return M.copy()   
    
    # ---- x full rounds (n agents vs ≥ n items) ----
    for _ in range(x):
        k = len(remaining)
        if k < n:
            break

        cost = build_cost(remaining)

        if cost.shape[0] <= cost.shape[1]:
            r, c = rectangular_assignment_exact(cost)
        else:
            r, c = rectangular_assignment_exact(cost.T)
            r, c = c, r

        # assign exactly one item to each agent (n matches)
        chosen_item_cols = set()
        for i, j in zip(r, c):
            i = int(i); j = int(j)
            item_idx = remaining[j]
            bundles[i].append(item_idx)
            chosen_item_cols.add(j)

        # remove chosen items from 'remaining' (sort descending so indices stay valid)
        for j in sorted(chosen_item_cols, reverse=True):
            remaining.pop(j)

    # ---- final leftover round (n agents vs y items) ----
    if y > 0 and remaining:
        cost = build_cost(remaining) 

        if cost.shape[0] <= cost.shape[1]:
            r, c = rectangular_assignment_exact(cost)
        else:
            rt, ct = rectangular_assignment_exact(cost.T)
            r, c = ct, rt

        for i, j in zip(r, c):
            i = int(i); j = int(j)
            item_idx = remaining[j]
            bundles[i].append(item_idx)
        remaining.clear()

    return Allocation(bundles)
