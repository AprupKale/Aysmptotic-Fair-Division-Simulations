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
from typing import List, Optional

from dists import RNG, ItemDist, UniformItem, BetaItem, TruncatedNormalItem, \
    KumaraswamyItem, TriangularItem, BatesItem, LogitNormalItem, TruncatedExponential01Item
from experiments import sweep_m

# --------------------------- Factories for item mixes ------------------------

def make_item_factory(name: str):
    name = (name or 'uniform').lower()
    if name == 'uniform':
        def factory(m, rng: RNG):
            return [UniformItem(0.0, 1.0) for _ in range(m)]
        return factory
    if name == 'beta':
        def factory(m, rng: RNG):
            return [BetaItem(2.0, 5.0) for _ in range(m)]
        return factory
    if name == 'normal':
        def factory(m, rng: RNG):
            return [TruncatedNormalItem(0.5, 0.05) for _ in range(m)]
        return factory
    if name == 'beta_uniform':
        def factory(m, rng: RNG):
            k = m // 2
            return [BetaItem(2.0,5.0)]*k + [UniformItem(0.0,1.0)]*(m-k)
        return factory
    if name == 'normal_uniform':
        def factory(m, rng: RNG):
            k = m // 2
            return [TruncatedNormalItem(0.5,0.05)]*k + [UniformItem(0.0,1.0)]*(m-k)
        return factory
    if name == 'mixed_demo':
        def factory(m, rng: RNG):
            palette: List[ItemDist] = [
                UniformItem(0.0, 1.0),
                BetaItem(2.0, 5.0),
                KumaraswamyItem(1.5, 3.0),
                TriangularItem(0.3),
                BatesItem(5),
                LogitNormalItem(0.0, 1.0),
                TruncatedExponential01Item(3.0),
                TruncatedNormalItem(0.5, 0.15)]
            items = [palette[i % len(palette)] for i in range(m)]
            return items
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
                    + [int(c * n * np.log(n)) for c in [10.0, 20.0, 30.0, 40.0]]

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
            seed=args.seed + d, 
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
