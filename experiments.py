# experiments.py
"""
Experiment runner and trend sweeps for the paper's simulations.

- Builds instances from per-item distributions (distinct D_j per item)
- Runs the two algorithms:
    * Large-regime: max-per-item or sampled max-per-item
    * Small-regime: matching (exact)    # (kept for future; not used by current plots)
- Averages over multiple trials
- Provides sweeps over m

Only numpy is required (SciPy optional via matching).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict
import numpy as np

from dists import ItemDist, UniformItem, RNG
from utils import Instance, Allocation, envy_metrics
from algorithms import alg_max_per_item, alg_matching, alg_max_per_item_sampled

# --------------------------- Instance construction ---------------------------

def make_instance(n: int, item_dists: List[ItemDist], rng: RNG, kind: str) -> Instance:
    cols = [d.sample_for_all_agents(n, rng) for d in item_dists]
    U = np.stack(cols, axis=1)
    return Instance(n=n, m=len(item_dists), U=U, kind=kind)

# --------------------------- Config + runner ---------------------------------

@dataclass
class ExperimentConfig:
    n: int
    m: int
    regime: str                                  # 'large' or 'small' (informational)
    num_samples: Optional[int] = None            # for large-regime sampled version
    kind: str = 'goods'                          # 'goods' or 'chores'
    item_dists: Optional[List[ItemDist]] = None  # if None -> all Uniform(0,1)
    trials: int = 50
    seed: Optional[int] = 0
    run_baseline: bool = True                    # run max-per-item
    run_sampled: bool = False                    # run sampled max-per-item

def _alloc_welfare(inst: Instance, A: Allocation) -> float:
    take = A.to_matrix(inst.n, inst.m) == 1
    s = float(inst.U[take].sum())
    # chores welfare as negative sum of disutilities -> maximize -sum
    return s if inst.kind == 'goods' else -s

def run_experiment(cfg: ExperimentConfig) -> Dict[str, float]:
    rng = RNG(cfg.seed)
    # Build item distributions; mixing supported
    if cfg.item_dists is None:
        item_dists = [UniformItem(0.0, 1.0) for _ in range(cfg.m)]
    else:
        assert len(cfg.item_dists) == cfg.m
        item_dists = cfg.item_dists

    worst_envy_ratios = {"max_per_item": 0.0, "max_per_item_sampled": 0.0}
    envious_agents_fraction = {"max_per_item": 0.0, "max_per_item_sampled": 0.0}
    welfares = {"max_per_item": 0.0, "max_per_item_sampled": 0.0}

    for _ in range(cfg.trials):
        inst = make_instance(cfg.n, item_dists, rng, kind=cfg.kind)

        if cfg.run_baseline:
            A = alg_max_per_item(inst)
            wer, frac = envy_metrics(inst, A)
            worst_envy_ratios["max_per_item"] += wer
            envious_agents_fraction["max_per_item"] += frac
            welfares["max_per_item"] += _alloc_welfare(inst, A)

        if cfg.run_sampled:
            if not isinstance(cfg.num_samples, int) or cfg.num_samples <= 0:
                raise ValueError("run_sampled=True requires a positive integer num_samples.")
            A = alg_max_per_item_sampled(inst, num_samples=cfg.num_samples)
            wer, frac = envy_metrics(inst, A)
            worst_envy_ratios["max_per_item_sampled"] += wer
            envious_agents_fraction["max_per_item_sampled"] += frac
            welfares["max_per_item_sampled"] += _alloc_welfare(inst, A)

    out: Dict[str, float] = {}
    if cfg.run_baseline:
        out.update({
            "worst_envy_ratio_max_per_item": worst_envy_ratios["max_per_item"] / cfg.trials,
            "envious_agents_fraction_max_per_item": envious_agents_fraction["max_per_item"] / cfg.trials,
            "welfare_max_per_item": welfares["max_per_item"] / cfg.trials,
        })
    if cfg.run_sampled:
        out.update({
            "worst_envy_ratio_max_per_item_sampled": worst_envy_ratios["max_per_item_sampled"] / cfg.trials,
            "envious_agents_fraction_max_per_item_sampled": envious_agents_fraction["max_per_item_sampled"] / cfg.trials,
            "welfare_max_per_item_sampled": welfares["max_per_item_sampled"] / cfg.trials,
        })
    return out

# --------------------------- Trend sweeps ------------------------------------

def sweep_m(n: int,
            m_list: List[int],
            kind: str,
            num_samples: Optional[int],
            trials: int,
            seed: Optional[int],
            run_baseline: bool,
            run_sampled: bool,
            item_dist_factory: Optional[Callable[[int, RNG], List[ItemDist]]] = None
            ) -> List[Tuple[int, Dict[str, float]]]:
    """
    Returns a list of (m, metrics_dict) pairs. Each metrics_dict only contains
    the keys for the variants requested (baseline and/or sampled).
    """
    rng = RNG(seed)
    results: List[Tuple[int, Dict[str, float]]] = []
    for m in m_list:
        if item_dist_factory is None:
            item_dists = [UniformItem(0.0, 1.0) for _ in range(m)]
        else:
            item_dists = item_dist_factory(m, rng)

        cfg = ExperimentConfig(
            n=n, m=m, regime='auto', kind=kind,
            item_dists=item_dists, trials=trials,
            seed=int(rng._g.integers(0, 10**9)),
            num_samples=num_samples,
            run_baseline=run_baseline,
            run_sampled=run_sampled
        )
        results.append((m, run_experiment(cfg)))
    return results
