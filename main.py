# main.py
"""
Main entry point to run experiments for the paper.

Example usage:
    - python main.py --mode sweep_m --n 20 --kind goods --trials 50 --seed 1 --mix uniform --outdir results

Produces JSON result files for:

    - Non-sampling sweep over m:
        results/results_n{n}_{mix}_{kind}_non_sampling.json

    - Sampling sweeps over m for k = floor(d log n), d in {1,2,5,10}:
        results/results_n{n}_{mix}_{kind}_s{s}.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from typing import List, Optional, Callable

from dists import RNG, ItemDist, UniformItem, BetaItem, TruncatedNormalItem, \
    KumaraswamyItem, TriangularItem, BatesItem, LogitNormalItem, TruncatedExponential01Item
from experiments import sweep_m

# --------------------------- Factories for item mixes ------------------------

def make_item_factory(name: str) -> Callable[[int, "RNG"], List["ItemDist"]]:
    """
    Each factory below samples fresh parameters per item using the provided rng.
    No shared object references; every list element is a new distribution instance.
    """

    def _uniform_item(rng):
        low = float(rng.uniform(0.0, 0.9))
        high = float(rng.uniform(low + 0.05, 1.0))
        return UniformItem(low, high)

    def _beta_item(rng):
        a = float(rng.uniform(0.5, 6.0))
        b = float(rng.uniform(0.5, 6.0))
        return BetaItem(a, b)

    def _normal_item(rng):
        mu = float(rng.uniform(0.05, 0.95))
        sigma = float(rng.uniform(0.02, 0.25))
        return TruncatedNormalItem(mu, sigma)

    def _kumaraswamy_item(rng):
        a = float(rng.uniform(0.5, 6.0))
        b = float(rng.uniform(0.5, 6.0))
        return KumaraswamyItem(a, b)

    def _triangular_item(rng):
        mode = float(rng.uniform(0.05, 0.95))
        return TriangularItem(mode)

    def _bates_item(rng):
        n = int(rng.integers(2, 11))
        return BatesItem(n)

    def _logitnormal_item(rng):
        mu = float(rng.uniform(-2.0, 2.0))
        sigma = float(rng.uniform(0.2, 1.0))
        return LogitNormalItem(mu, sigma)

    def _truncexp01_item(rng):
        lam = float(rng.uniform(0.5, 10.0))
        return TruncatedExponential01Item(lam)

    name = (name or 'uniform').lower()

    if name == 'uniform':
        def factory(m, rng):
            return [_uniform_item(rng) for _ in range(m)]
        return factory

    if name == 'beta':
        def factory(m, rng):
            return [_beta_item(rng) for _ in range(m)]
        return factory

    if name == 'normal':
        def factory(m, rng):
            return [_normal_item(rng) for _ in range(m)]
        return factory

    if name == 'beta_uniform':
        def factory(m, rng):
            k = m // 2
            items = [_beta_item(rng) for _ in range(k)]
            items += [_uniform_item(rng) for _ in range(m - k)]
            return items
        return factory

    if name == 'normal_uniform':
        def factory(m, rng):
            k = m // 2
            items = [_normal_item(rng) for _ in range(k)]
            items += [_uniform_item(rng) for _ in range(m - k)]
            return items
        return factory

    if name == 'mixed_demo':
        palette_fns = [
            _uniform_item,
            _beta_item,
            _kumaraswamy_item,
            _triangular_item,
            _bates_item,
            _logitnormal_item,
            _truncexp01_item,
            _normal_item,
        ]
        def factory(m, rng):
            L = len(palette_fns)
            return [palette_fns[i % L](rng) for i in range(m)]
        return factory

    raise ValueError(f"Unknown mix name: {name}")

# --------------------------- CLI --------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['sweep_m'], default='sweep_m')
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--m_list', type=int, nargs='*',
                   help='List of m values; if omitted, a default grid is used.')
    p.add_argument('--kind', choices=['goods','chores'], default='goods')
    p.add_argument('--trials', type=int, default=50)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--mix', type=str, default='uniform')
    p.add_argument('--outdir', type=str, default='results',
                   help='Directory to write JSON result files.')
    return p.parse_args()

# --------------------------- Main -------------------------------------------

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = RNG(args.seed)
    factory = make_item_factory(args.mix)

    # Default sweep grid for m (both linear and n log n scales)
    if not args.m_list:
        n = args.n
        args.m_list = [int(c * n) for c in [10.0, 20.0, 50.0, 100.0]] \
                    + [int(c * n * np.log(n)) for c in [20.0, 50.0, 100.0, 200.0]]

    # ---- Non-sampling (baseline) file ----
    res_ns = sweep_m(
        n=args.n,
        m_list=args.m_list,
        kind=args.kind,
        num_samples=None,
        trials=args.trials,
        seed=args.seed,
        run_baseline=True,
        run_sampled=False,
        item_dist_factory=factory
    )
    obj_ns = {str(int(m)): r for m, r in res_ns}
    ns_path = outdir / f'results_n{args.n}_{args.mix}_{args.kind}_non_sampling.json'
    with ns_path.open('w') as f:
        json.dump(obj_ns, f, sort_keys=True)
    print(f'Wrote {ns_path}')

    # ---- Sampling files for k = floor(d log n) ----
    logn = float(np.log(args.n))
    for d in (1, 2, 5, 10):
        s = max(1, int(d * logn))
        if s > args.n:
            break
        res_s = sweep_m(
            n=args.n,
            m_list=args.m_list,
            kind=args.kind,
            num_samples=s,
            trials=args.trials,
            seed=args.seed, 
            run_baseline=False,
            run_sampled=True,
            item_dist_factory=factory
        )
        obj_s = {str(int(m)): r for m, r in res_s}
        spath = outdir / f'results_n{args.n}_{args.mix}_{args.kind}_s{s}.json'
        with spath.open('w') as f:
            json.dump(obj_s, f, sort_keys=True)
        print(f'Wrote {spath}')

if __name__ == '__main__':
    main()
