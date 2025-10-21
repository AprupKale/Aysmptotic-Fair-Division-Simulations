"""
Experiment runner and trend sweeps for the paper's simulations.

- Builds instances from per-item distributions (distinct D_j per item)
- Runs the two algorithms:
    * Large-regime: max-per-item
    * Small-regime: matching (exact)
- Averages over multiple trials
- Provides sweeps over m and over n

Only numpy is required (SciPy optional via matching).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict
import numpy as np

from dists import ItemDist, UniformItem, RNG
from utils import Instance, Allocation
from algorithms import alg_max_per_item, envy_free, alg_matching

# Instance construction ---------------------------------------

def make_instance(n: int, item_dists: List[ItemDist], rng: RNG, kind: str) -> Instance:
    cols = [d.sample_for_all_agents(n, rng) for d in item_dists]
    U = np.stack(cols, axis=1)
    return Instance(n=n, m=len(item_dists), U=U, kind=kind)

# Config + runner ---------------------------------------------

@dataclass
class ExperimentConfig:
    n: int
    m: int
    regime: str                 # 'large' or 'small' (informational)
    kind: str = 'goods'         # 'goods' or 'chores'
    item_dists: Optional[List[ItemDist]] = None  # if None -> all Uniform(0,1)
    trials: int = 50
    seed: Optional[int] = 0


def run_experiment(cfg: ExperimentConfig) -> Dict[str, float]:
    rng = RNG(cfg.seed)
    # Build item distributions; mixing supported
    if cfg.item_dists is None:
        item_dists = [UniformItem(0.0, 1.0) for _ in range(cfg.m)]
    else:
        assert len(cfg.item_dists) == cfg.m
        item_dists = cfg.item_dists

    ef_rates = {"max_per_item": 0.0, "matching": 0.0}
    mean_envy = {"max_per_item": 0.0, "matching": 0.0}
    mean_welf = {"max_per_item": 0.0, "matching": 0.0}

    for _ in range(cfg.trials):
        inst = make_instance(cfg.n, item_dists, rng, kind=cfg.kind)

        # Large-regime: maximum-utility per item
        A1 = alg_max_per_item(inst)
        ef1, e1 = envy_free(inst, A1)
        w1 = float(inst.U[A1.to_matrix(inst.n, inst.m)==1].sum()) if cfg.kind=='goods' else float(-inst.U[A1.to_matrix(inst.n, inst.m)==1].sum())
        ef_rates["max_per_item"] += 1.0 if ef1 else 0.0
        mean_envy["max_per_item"] += e1
        mean_welf["max_per_item"] += w1

        # Small-regime: matching
        A2 = alg_matching(inst)
        ef2, e2 = envy_free(inst, A2)
        w2 = float(inst.U[A2.to_matrix(inst.n, inst.m)==1].sum()) if cfg.kind=='goods' else float(-inst.U[A2.to_matrix(inst.n, inst.m)==1].sum())
        ef_rates["matching"] += 1.0 if ef2 else 0.0
        mean_envy["matching"] += e2
        mean_welf["matching"] += w2

    out = {
        "ef_rate_max_per_item": ef_rates["max_per_item"] / cfg.trials,
        "ef_rate_matching": ef_rates["matching"] / cfg.trials,
        "mean_max_envy_max_per_item": mean_envy["max_per_item"] / cfg.trials,
        "mean_max_envy_matching": mean_envy["matching"] / cfg.trials,
        "mean_welfare_max_per_item": mean_welf["max_per_item"] / cfg.trials,
        "mean_welfare_matching": mean_welf["matching"] / cfg.trials,
    }
    return out

# Trend sweeps -------------------------------------------------

def sweep_m(n: int, m_list: List[int], kind: str, trials: int, seed: Optional[int], item_dist_factory: Optional[Callable[[int, RNG], List[ItemDist]]] = None) -> List[Tuple[int, Dict[str, float]]]:
    rng = RNG(seed)
    results = []
    for m in m_list:
        if item_dist_factory is None:
            item_dists = [UniformItem(0.0, 1.0) for _ in range(m)]
        else:
            item_dists = item_dist_factory(m, rng)
        cfg = ExperimentConfig(n=n, m=m, regime='auto', kind=kind, item_dists=item_dists, trials=trials, seed=rng._g.integers(0, 10**9))
        results.append((m, run_experiment(cfg)))
    return results


def sweep_n(n_list: List[int], m: int, kind: str, trials: int, seed: Optional[int], item_dist_factory: Optional[Callable[[int, RNG], List[ItemDist]]] = None) -> List[Tuple[int, Dict[str, float]]]:
    rng = RNG(seed)
    results = []
    for n in n_list:
        if item_dist_factory is None:
            base_dists = [UniformItem(0.0, 1.0) for _ in range(m)]
        else:
            base_dists = item_dist_factory(m, rng)
        cfg = ExperimentConfig(n=n, m=m, regime='auto', kind=kind, item_dists=base_dists, trials=trials, seed=rng._g.integers(0, 10**9))
        results.append((n, run_experiment(cfg)))
    return results
