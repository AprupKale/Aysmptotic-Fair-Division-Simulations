#!/usr/bin/env python3
"""
Visualization script for fair division experiments.
Generates comprehensive plots from experiment results.

Usage:
    python visualize_results.py --input results/all_results.csv --output plots/
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess experiment results."""
    df = pd.read_csv(csv_path)
    df['n'] = pd.to_numeric(df['n'], errors='coerce')
    df['m'] = pd.to_numeric(df['m'], errors='coerce')
    df['m_over_n'] = df['m'] / df['n']
    df['n_over_m'] = df['n'] / df['m']
    df['regime'] = df.apply(lambda row: 
        'Large (m>>n)' if row['m_over_n'] > 10 else 
        'Small (n>>m)' if row['n_over_m'] > 10 else 
        'Balanced', axis=1)
    return df

def plot_ef_rate_vs_m(df: pd.DataFrame, output_dir: Path):
    """Plot envy-free rate as m increases (fixed n)."""
    sweep_m = df[df['sweep_type'] == 'm'].copy()
    
    for kind in sweep_m['kind'].unique():
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Envy-Free Rate vs Number of Items (m) - {kind.capitalize()}', 
                     fontsize=16, fontweight='bold')
        
        for idx, mix in enumerate(sweep_m['mix'].unique()[:4]):
            ax = axes[idx // 2, idx % 2]
            subset = sweep_m[(sweep_m['kind'] == kind) & (sweep_m['mix'] == mix)]
            
            for n_val in sorted(subset['n'].unique()):
                n_data = subset[subset['n'] == n_val].sort_values('m')
                ax.plot(n_data['m'], n_data['ef_rate_max_per_item'], 
                       marker='o', label=f'Max-per-item (n={n_val})', linewidth=2)
                ax.plot(n_data['m'], n_data['ef_rate_matching'], 
                       marker='s', linestyle='--', label=f'Matching (n={n_val})', linewidth=2)
            
            ax.set_xlabel('Number of items (m)', fontsize=12)
            ax.set_ylabel('Envy-free rate', fontsize=12)
            ax.set_title(f'Mix: {mix}', fontsize=13)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'ef_rate_vs_m_{kind}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: ef_rate_vs_m_{kind}.png")

def plot_ef_rate_vs_n(df: pd.DataFrame, output_dir: Path):
    """Plot envy-free rate as n increases (fixed m)."""
    sweep_n = df[df['sweep_type'] == 'n'].copy()
    
    for kind in sweep_n['kind'].unique():
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Envy-Free Rate vs Number of Agents (n) - {kind.capitalize()}', 
                     fontsize=16, fontweight='bold')
        
        for idx, mix in enumerate(sweep_n['mix'].unique()[:4]):
            ax = axes[idx // 2, idx % 2]
            subset = sweep_n[(sweep_n['kind'] == kind) & (sweep_n['mix'] == mix)]
            
            for m_val in sorted(subset['m'].unique()):
                m_data = subset[subset['m'] == m_val].sort_values('n')
                ax.plot(m_data['n'], m_data['ef_rate_max_per_item'], 
                       marker='o', label=f'Max-per-item (m={m_val})', linewidth=2)
                ax.plot(m_data['n'], m_data['ef_rate_matching'], 
                       marker='s', linestyle='--', label=f'Matching (m={m_val})', linewidth=2)
            
            ax.set_xlabel('Number of agents (n)', fontsize=12)
            ax.set_ylabel('Envy-free rate', fontsize=12)
            ax.set_title(f'Mix: {mix}', fontsize=13)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'ef_rate_vs_n_{kind}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: ef_rate_vs_n_{kind}.png")

def plot_regime_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare algorithms across regimes (large vs small vs balanced)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Algorithm Performance by Regime', fontsize=16, fontweight='bold')
    
    metrics = [
        ('ef_rate_max_per_item', 'EF Rate: Max-per-item'),
        ('ef_rate_matching', 'EF Rate: Matching'),
        ('mean_max_envy_max_per_item', 'Max Envy: Max-per-item'),
        ('mean_max_envy_matching', 'Max Envy: Matching'),
        ('mean_welfare_max_per_item', 'Welfare: Max-per-item'),
        ('mean_welfare_matching', 'Welfare: Matching'),
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        # Aggregate by regime and kind
        grouped = df.groupby(['regime', 'kind'])[metric].mean().reset_index()
        
        for kind in grouped['kind'].unique():
            kind_data = grouped[grouped['kind'] == kind]
            ax.bar([f"{r}\n({kind})" for r in kind_data['regime']], 
                   kind_data[metric], 
                   alpha=0.7, 
                   label=kind.capitalize())
        
        ax.set_ylabel(title, fontsize=11)
        ax.set_xlabel('Regime', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        if 'rate' in metric.lower():
            ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'regime_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: regime_comparison.png")

def plot_welfare_vs_fairness(df: pd.DataFrame, output_dir: Path):
    """Scatter plot: welfare vs envy-free rate trade-off."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Welfare vs Fairness Trade-off', fontsize=16, fontweight='bold')
    
    algorithms = [
        ('max_per_item', 'Max-per-item', 'o'),
        ('matching', 'Matching', 's')
    ]
    
    for ax_idx, kind in enumerate(['goods', 'chores']):
        ax = axes[ax_idx]
        kind_data = df[df['kind'] == kind]
        
        for alg_name, alg_label, marker in algorithms:
            ef_col = f'ef_rate_{alg_name}'
            welfare_col = f'mean_welfare_{alg_name}'
            
            ax.scatter(kind_data[ef_col], kind_data[welfare_col], 
                      alpha=0.6, s=50, marker=marker, label=alg_label)
        
        ax.set_xlabel('Envy-free rate', fontsize=12)
        ax.set_ylabel('Mean welfare', fontsize=12)
        ax.set_title(f'{kind.capitalize()}', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'welfare_vs_fairness.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: welfare_vs_fairness.png")

def plot_envy_heatmap(df: pd.DataFrame, output_dir: Path):
    """Heatmap of mean maximum envy across (n, m) combinations."""
    for kind in df['kind'].unique():
        for alg in ['max_per_item', 'matching']:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            kind_data = df[df['kind'] == kind].copy()
            envy_col = f'mean_max_envy_{alg}'
            
            # Create pivot table
            pivot = kind_data.pivot_table(
                values=envy_col, 
                index='n', 
                columns='m', 
                aggfunc='mean'
            )
            
            sns.heatmap(pivot, annot=False, fmt='.3f', cmap='RdYlGn_r', 
                       ax=ax, cbar_kws={'label': 'Mean Max Envy'})
            
            ax.set_title(f'Mean Maximum Envy - {alg.replace("_", " ").title()} ({kind.capitalize()})', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Number of items (m)', fontsize=12)
            ax.set_ylabel('Number of agents (n)', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'envy_heatmap_{alg}_{kind}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: envy_heatmap_{alg}_{kind}.png")

def plot_mix_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare different item distributions (mixes)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Effect of Item Distribution on Algorithm Performance', 
                 fontsize=16, fontweight='bold')
    
    metrics = [
        ('ef_rate_max_per_item', 'EF Rate: Max-per-item'),
        ('ef_rate_matching', 'EF Rate: Matching'),
        ('mean_welfare_max_per_item', 'Welfare: Max-per-item'),
        ('mean_welfare_matching', 'Welfare: Matching'),
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        grouped = df.groupby(['mix', 'kind'])[metric].mean().reset_index()
        
        x = np.arange(len(grouped['mix'].unique()))
        width = 0.35
        
        for kind_idx, kind in enumerate(['goods', 'chores']):
            kind_data = grouped[grouped['kind'] == kind].sort_values('mix')
            ax.bar(x + kind_idx * width, kind_data[metric], width, 
                   alpha=0.7, label=kind.capitalize())
        
        ax.set_ylabel(title, fontsize=11)
        ax.set_xlabel('Item Distribution', fontsize=11)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(grouped['mix'].unique(), rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        if 'rate' in metric.lower():
            ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mix_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: mix_comparison.png")

def plot_ratio_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze performance as a function of m/n ratio."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance vs m/n Ratio (log scale)', fontsize=16, fontweight='bold')
    
    for kind_idx, kind in enumerate(['goods', 'chores']):
        kind_data = df[df['kind'] == kind].copy()
        kind_data = kind_data[kind_data['m_over_n'] > 0]  # Valid ratios only
        
        # EF rates
        ax = axes[kind_idx, 0]
        for mix in kind_data['mix'].unique():
            mix_data = kind_data[kind_data['mix'] == mix].sort_values('m_over_n')
            ax.plot(mix_data['m_over_n'], mix_data['ef_rate_max_per_item'], 
                   marker='o', label=f'{mix} (max-per-item)', alpha=0.7)
            ax.plot(mix_data['m_over_n'], mix_data['ef_rate_matching'], 
                   marker='s', linestyle='--', label=f'{mix} (matching)', alpha=0.7)
        
        ax.set_xscale('log')
        ax.set_xlabel('m/n ratio (log scale)', fontsize=11)
        ax.set_ylabel('Envy-free rate', fontsize=11)
        ax.set_title(f'{kind.capitalize()} - EF Rate', fontsize=12)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.axvline(x=1, color='red', linestyle=':', alpha=0.5, label='m=n')
        
        # Welfare
        ax = axes[kind_idx, 1]
        for mix in kind_data['mix'].unique():
            mix_data = kind_data[kind_data['mix'] == mix].sort_values('m_over_n')
            ax.plot(mix_data['m_over_n'], mix_data['mean_welfare_max_per_item'], 
                   marker='o', label=f'{mix} (max-per-item)', alpha=0.7)
            ax.plot(mix_data['m_over_n'], mix_data['mean_welfare_matching'], 
                   marker='s', linestyle='--', label=f'{mix} (matching)', alpha=0.7)
        
        ax.set_xscale('log')
        ax.set_xlabel('m/n ratio (log scale)', fontsize=11)
        ax.set_ylabel('Mean welfare', fontsize=11)
        ax.set_title(f'{kind.capitalize()} - Welfare', fontsize=12)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=1, color='red', linestyle=':', alpha=0.5, label='m=n')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ratio_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: ratio_analysis.png")

def generate_summary_stats(df: pd.DataFrame, output_dir: Path):
    """Generate summary statistics table."""
    summary = []
    
    for kind in df['kind'].unique():
        for alg in ['max_per_item', 'matching']:
            kind_data = df[df['kind'] == kind]
            
            stats = {
                'Kind': kind.capitalize(),
                'Algorithm': alg.replace('_', ' ').title(),
                'Mean EF Rate': f"{kind_data[f'ef_rate_{alg}'].mean():.3f}",
                'Std EF Rate': f"{kind_data[f'ef_rate_{alg}'].std():.3f}",
                'Mean Welfare': f"{kind_data[f'mean_welfare_{alg}'].mean():.2f}",
                'Std Welfare': f"{kind_data[f'mean_welfare_{alg}'].std():.2f}",
                'Mean Max Envy': f"{kind_data[f'mean_max_envy_{alg}'].mean():.4f}",
                'Std Max Envy': f"{kind_data[f'mean_max_envy_{alg}'].std():.4f}",
            }
            summary.append(stats)
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("="*70 + "\n")
    print(f"Saved: summary_statistics.csv")

def main():
    parser = argparse.ArgumentParser(description='Visualize fair division experiment results')
    parser.add_argument('--input', type=str, default='results/all_results.csv',
                       help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='plots/',
                       help='Output directory for plots')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {args.input}...")
    df = load_data(args.input)
    print(f"Loaded {len(df)} experiment results\n")
    
    print("Generating visualizations...")
    print("-" * 70)
    
    plot_ef_rate_vs_m(df, output_dir)
    plot_ef_rate_vs_n(df, output_dir)
    plot_regime_comparison(df, output_dir)
    plot_welfare_vs_fairness(df, output_dir)
    plot_envy_heatmap(df, output_dir)
    plot_mix_comparison(df, output_dir)
    plot_ratio_analysis(df, output_dir)
    generate_summary_stats(df, output_dir)
    
    print("-" * 70)
    print(f"\nAll visualizations saved to {output_dir}/")

if __name__ == '__main__':
    main()
