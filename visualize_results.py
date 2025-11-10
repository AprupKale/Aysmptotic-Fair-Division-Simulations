#!/usr/bin/env python3
"""
Visualizations for fair-division experiments.

Features
- Non-sampling + sampling plots (envy metrics)
- Comparison plots
- Welfare ratio (sampling / non-sampling) per s value
- Style presets: paper / talk / dark

Usage:
    python visualize_results.py --input_dir results/ --output plots/ \
        --kind goods --mix uniform --n 20 --style talk
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ----------------------------- Styling ---------------------------------------

CB_PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#F0E442", "#56B4E9", "#E69F00", "#000000"
]

def set_style(mode: str = "paper"):
    """
    mode: 'paper' (default), 'talk', or 'dark'
    """
    plt.rcParams.update({
        "figure.figsize": (12, 7),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "axes.labelpad": 8,
        "legend.frameon": False,
        "legend.borderaxespad": 0.6,
        "lines.linewidth": 2.5,
        "lines.markersize": 7,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "font.size": 11,
    })
    if mode == "talk":
        plt.rcParams.update({"font.size": 13, "lines.linewidth": 3.0, "lines.markersize": 8})
    if mode == "dark":
        plt.style.use("dark_background")
        plt.rcParams.update({"grid.color": "#666666", "axes.edgecolor": "#CCCCCC"})
    # set default color cycle
    from cycler import cycler
    plt.rcParams["axes.prop_cycle"] = cycler(color=CB_PALETTE)

def savefig_both(outdir: Path, fname: str):
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{fname}.png"
    svg = outdir / f"{fname}.svg"
    plt.savefig(png, transparent=False)
    plt.savefig(svg, transparent=True)
    print(f"Saved: {png.name}, {svg.name}")

# ----------------------------- IO --------------------------------------------

def load_json(path: Path) -> Dict:
    with path.open('r') as f:
        return json.load(f)

# ----------------------------- Plots -----------------------------------------

def plot_non_sampling_envy_metrics(data: Dict, output_dir: Path, kind: str, mix: str, n: int):
    m_values = sorted(int(m) for m in data.keys())
    wer = [data[str(m)]['worst_envy_ratio_max_per_item'] for m in m_values]
    frac = [data[str(m)]['envious_agents_fraction_max_per_item'] for m in m_values]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 9))
    fig.suptitle(f'Non-Sampling: {kind.capitalize()}, {mix}, n={n}')

    ax1.plot(m_values, wer, marker='o')
    ax1.set_ylabel('Worst envy ratio')
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax2.plot(m_values, frac, marker='s')
    ax2.set_xlabel('Number of items (m)')
    ax2.set_ylabel('Fraction with envy')
    ax2.set_ylim(-0.02, 1.02)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))

    plt.tight_layout()
    savefig_both(output_dir, f'non_sampling_{kind}_{mix}_n{n}')
    plt.close()


def plot_sampling_envy_metrics(data_dict: Dict[int, Dict], output_dir: Path, kind: str, mix: str, n: int):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 9))
    fig.suptitle(f'Sampling: {kind.capitalize()}, {mix}, n={n}')

    for idx, s_val in enumerate(sorted(data_dict.keys())):
        data = data_dict[s_val]
        m_values = sorted(int(m) for m in data.keys())
        wer = [data[str(m)]['worst_envy_ratio_max_per_item_sampled'] for m in m_values]
        frac = [data[str(m)]['envious_agents_fraction_max_per_item_sampled'] for m in m_values]

        ax1.plot(m_values, wer, marker='o', label=f's={s_val}')
        ax2.plot(m_values, frac, marker='s', label=f's={s_val}')

    ax1.set_ylabel('Worst envy ratio')
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax1.legend(ncol=2, fontsize=10)

    ax2.set_xlabel('Number of items (m)')
    ax2.set_ylabel('Fraction with envy')
    ax2.set_ylim(-0.02, 1.02)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax2.legend(ncol=2, fontsize=10)

    plt.tight_layout()
    savefig_both(output_dir, f'sampling_{kind}_{mix}_n{n}')
    plt.close()


def plot_sampling_vs_non_sampling_comparison(non_sampling_data: Dict,
                                             sampling_data_dict: Dict[int, Dict],
                                             output_dir: Path, kind: str, mix: str, n: int):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 9))
    fig.suptitle(f'Comparison: {kind.capitalize()}, {mix}, n={n}')

    m_values = sorted(int(m) for m in non_sampling_data.keys())
    wer_ns = [non_sampling_data[str(m)]['worst_envy_ratio_max_per_item'] for m in m_values]
    frac_ns = [non_sampling_data[str(m)]['envious_agents_fraction_max_per_item'] for m in m_values]

    # heavy baseline
    ax1.plot(m_values, wer_ns, marker='o', color="#222222", linewidth=3.5, label='Non-sampling')
    ax2.plot(m_values, frac_ns, marker='o', color="#222222", linewidth=3.5, label='Non-sampling')

    for s_val in sorted(sampling_data_dict.keys()):
        data = sampling_data_dict[s_val]
        m_s = sorted(int(m) for m in data.keys())
        wer = [data[str(m)]['worst_envy_ratio_max_per_item_sampled'] for m in m_s]
        frac = [data[str(m)]['envious_agents_fraction_max_per_item_sampled'] for m in m_s]

        ax1.plot(m_s, wer, marker='^', label=f's={s_val}', alpha=0.9)
        ax2.plot(m_s, frac, marker='^', label=f's={s_val}', alpha=0.9)

    ax1.set_ylabel('Worst envy ratio')
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax1.legend(ncol=2, fontsize=10)

    ax2.set_xlabel('Number of items (m)')
    ax2.set_ylabel('Fraction with envy')
    ax2.set_ylim(-0.02, 1.02)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax2.legend(ncol=2, fontsize=10)

    plt.tight_layout()
    savefig_both(output_dir, f'comparison_{kind}_{mix}_n{n}')
    plt.close()


def plot_welfare_ratio(non_sampling_data: Dict,
                       sampling_data_dict: Dict[int, Dict],
                       output_dir: Path, kind: str, mix: str, n: int):
    base_w = {int(m): non_sampling_data[m]['welfare_max_per_item']
              for m in non_sampling_data.keys()}

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.suptitle(f'Welfare Ratio (Sampling / Non-sampling): {kind.capitalize()}, {mix}, n={n}')

    for s_val in sorted(sampling_data_dict.keys()):
        data_s = sampling_data_dict[s_val]
        samp_w = {int(m): data_s[m]['welfare_max_per_item_sampled'] for m in data_s.keys()}

        common_ms = sorted(set(base_w.keys()).intersection(samp_w.keys()))
        if not common_ms:
            continue

        ratio = []
        for m in common_ms:
            if kind == 'goods':
                denom = base_w[m]
                num = samp_w[m]
            else:  # chores
                denom = -samp_w[m]
                num = -base_w[m]
            r = np.nan if denom == 0 else (num / denom)
            ratio.append(r)

        ax.plot(common_ms, ratio, marker='o', label=f's={s_val}', alpha=0.95)

        finite = [x for x in ratio if np.isfinite(x)]
        if finite:
            print(f"[WELFARE RATIO] n={n}, {mix}/{kind}, s={s_val} | "
                  f"mean={np.mean(finite):.4f}, std={np.std(finite):.4f}, "
                  f"min={np.min(finite):.4f}, max={np.max(finite):.4f}")

    ax.axhline(1.0, linestyle='--', linewidth=1.5, color='#888888', alpha=0.8)
    ax.set_xlabel('Number of items (m)')
    ax.set_ylabel('Welfare ratio: sampled / non-sampling')
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(ncol=2, fontsize=10)
    plt.tight_layout()
    savefig_both(output_dir, f'welfare_ratio_{kind}_{mix}_n{n}')
    plt.close()

# ----------------------------- Main ------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Visualize fair division experiment results')
    parser.add_argument('--input_dir', type=str, default='results/',
                        help='Directory containing JSON result files')
    parser.add_argument('--output', type=str, default='plots/',
                        help='Output directory for plots')
    parser.add_argument('--kind', type=str, default='goods', choices=['goods', 'chores'],
                        help='Type of items')
    parser.add_argument('--mix', type=str, default='uniform',
                        help='Item distribution mix')
    parser.add_argument('--n', type=int, default=20,
                        help='Number of agents')
    parser.add_argument('--style', type=str, default='paper', choices=['paper','talk','dark'],
                        help='Visual style preset')
    args = parser.parse_args()

    set_style(args.style)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input_dir)

    non_sampling_path = input_dir / f'results_n{args.n}_{args.mix}_{args.kind}_non_sampling.json'
    non_sampling_data: Optional[Dict] = None
    sampling_data_dict: Dict[int, Dict] = {}

    if non_sampling_path.exists():
        non_sampling_data = load_json(non_sampling_path)
        print(f"Loaded non-sampling data: {non_sampling_path.name}")
        plot_non_sampling_envy_metrics(non_sampling_data, output_dir, args.kind, args.mix, args.n)
    else:
        print(f"Warning: Non-sampling file not found: {non_sampling_path.name}")

    log_n = np.log(args.n)
    s_candidates = []
    for d in (1, 2, 5, 10):
        s = max(1, int(d * log_n))
        if s > args.n:
            break
        s_candidates.append(s)
    for s in s_candidates:
        p = input_dir / f'results_n{args.n}_{args.mix}_{args.kind}_s{s}.json'
        if p.exists():
            sampling_data_dict[s] = load_json(p)
            print(f"Loaded sampling data for s={s}: {p.name}")
        else:
            print(f"Warning: Sampling file not found for s={s}: {p.name}")

    if sampling_data_dict:
        plot_sampling_envy_metrics(sampling_data_dict, output_dir, args.kind, args.mix, args.n)
        if non_sampling_data:
            plot_sampling_vs_non_sampling_comparison(non_sampling_data, sampling_data_dict,
                                                     output_dir, args.kind, args.mix, args.n)
            plot_welfare_ratio(non_sampling_data, sampling_data_dict,
                               output_dir, args.kind, args.mix, args.n)

    print(f"All visualizations saved to {output_dir}/")

if __name__ == '__main__':
    main()
