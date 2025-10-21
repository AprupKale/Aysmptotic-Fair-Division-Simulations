"""
Main entry point to run experiments for the paper.

Examples
--------
# Large-regime (goods), mixed item distributions, sweep over m
python main.py --mode sweep_m --n 20 --m_list 200 400 600 800 --kind goods --trials 50 --mix beta_uniform

# Small-regime (chores), fixed (n,m)
python main.py --mode single --n 20 --m 300 --kind chores --trials 50 --mix uniform

Mix options
-----------
- uniform: all items U(0,1)
- beta_uniform: half Beta(2,5), half U(0,1)
- normal_uniform: half TruncatedNormal(mu=0.5,sigma=0.05), half U(0,1)
- beta: all items Beta(2,5)
- normal: all items TruncatedNormal(mu=0.5,sigma=0.05)
- mixed_demo: mix of different distributions
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from pyparsing import List, Optional

from dists import RNG, ItemDist, UniformItem, BetaItem, TruncatedNormalItem, \
    KumaraswamyItem, TriangularItem, BatesItem, LogitNormalItem, TruncatedExponential01Item
from experiments import ExperimentConfig, run_experiment, sweep_m, sweep_n

# Factories for item mixes ------------------------------------

def make_item_factory(name: str):
    name = (name or 'uniform').lower()
    if name == 'uniform':
        def factory(m, rng: RNG):
            return [UniformItem(0.0, 1.0) for _ in range(m)]
        return factory
    if name == 'beta':
        def factory(m, rng: RNG):
            k = m // 2
            return [BetaItem(2.0,5.0) for _ in range(m)]
        return factory
    if name == 'normal':
        def factory(m, rng: RNG):
            k = m // 2
            return [TruncatedNormalItem(0.5,0.05) for _ in range(m)]
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

# CLI ---------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['single','sweep_m','sweep_n'], required=True)
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--m', type=int, default=100)
    p.add_argument('--m_list', type=int, nargs='*')
    p.add_argument('--n_list', type=int, nargs='*')
    p.add_argument('--kind', choices=['goods','chores'], default='goods')
    p.add_argument('--trials', type=int, default=50)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--mix', type=str, default='uniform')
    return p.parse_args()


def main():
    args = parse_args()
    rng = RNG(args.seed)
    factory = make_item_factory(args.mix)

    if args.mode == 'single':
        item_dists = factory(args.m, rng)
        cfg = ExperimentConfig(n=args.n, m=args.m, regime='auto', kind=args.kind,
                               item_dists=item_dists, trials=args.trials, seed=args.seed)
        out = run_experiment(cfg)
        print(json.dumps(out))
        return

    if args.mode == 'sweep_m':
        if not args.m_list:
            args.m_list = [int(c * args.n) for c in [10.0, 20.0, 30.0, 40.0]] \
                + [int(c * args.n * np.log(max(3, args.n))) for c in [1.0, 5.0, 10.0, 20.0]]
        res = sweep_m(n=args.n, m_list=args.m_list, kind=args.kind, trials=args.trials, seed=args.seed, item_dist_factory=factory)
        out = {int(m): r for m, r in res}
        print(json.dumps(out))
        return

    if args.mode == 'sweep_n':
        if not args.n_list:
            args.n_list = [int(c * args.m) for c in [10.0, 20.0, 30.0, 40.0]] \
                + [int((c * args.m) / np.log(max(3, args.n))) for c in [1.0, 5.0, 10.0, 20.0]]
        res = sweep_n(n_list=args.n_list, m=args.m, kind=args.kind, trials=args.trials, seed=args.seed, item_dist_factory=factory)
        out = {int(n): r for n, r in res}
        print(json.dumps(out))
        return


if __name__ == '__main__':
    main()
