#!/bin/bash
#SBATCH --job-name=fair_division_simulations
#SBATCH --output=logs/experiment_%A_%a.out
#SBATCH --error=logs/experiment_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-32

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CS2109S

# Create output directories
mkdir -p results logs

# Define experiment parameters
N_VALUES=(10 25 50 100)
KINDS=("goods" "chores")
MIXES=("beta_uniform" "normal_uniform" "mixed_demo")
TRIALS=10

# Calculate which experiment to run based on SLURM_ARRAY_TASK_ID
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}

# Decode task ID into parameters
IDX=$((TASK_ID - 1))
N_IDX=$((IDX % 4))
KIND_IDX=$(((IDX / 4) % 2))
MIX_IDX=$((IDX / 8))

N=${N_VALUES[$N_IDX]}
KIND=${KINDS[$KIND_IDX]}
MIX=${MIXES[$MIX_IDX]}

echo "Running experiment: n=$N, kind=$KIND, mix=$MIX"

# ============================================
# SWEEP M (varying number of items)
# ============================================
OUTPUT_M="results/sweep_m_n${N}_${KIND}_${MIX}.json"

echo "Running sweep_m with m_list: $M_LIST"
python3 main.py \
    --mode sweep_m \
    --n $N \
    --kind $KIND \
    --mix $MIX \
    --trials $TRIALS \
    --seed 42 \
    > $OUTPUT_M

echo "Completed sweep_m -> $OUTPUT_M"

# ============================================
# Convert JSON to CSV for easier analysis
# ============================================
python3 << 'PYEOF'
import json
import csv
import sys
from pathlib import Path

results_dir = Path("results")
csv_rows = []

# Process all JSON files
for json_file in results_dir.glob("*.json"):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Parse filename: sweep_m_n{N}_{kind}_{mix}.json
        parts = json_file.stem.split('_')
        sweep_type = parts[1]  # 'm'
        
        # Extract metadata from filename
        n_fixed = None
        for part in parts[2:]:
            if part.startswith('n'):
                n_fixed = part[1:]
                break
        
        kind = parts[3] if len(parts) > 3 else 'unknown'
        mix = parts[4] if len(parts) > 4 else 'unknown'
        
        # Each key in JSON is a parameter value (m)
        for param_val, metrics in data.items():
            row = {
                'sweep_type': sweep_type,
                'n': n_fixed,
                'm': param_val,
                'kind': kind,
                'mix': mix,
                **metrics
            }
            csv_rows.append(row)
    except Exception as e:
        print(f"Error processing {json_file}: {e}", file=sys.stderr)

# Write consolidated CSV
if csv_rows:
    csv_file = results_dir / "all_results.csv"
    fieldnames = ['sweep_type', 'n', 'm', 'kind', 'mix', 
                  'ef_rate_max_per_item', 'ef_rate_matching',
                  'mean_max_envy_max_per_item', 'mean_max_envy_matching',
                  'mean_welfare_max_per_item', 'mean_welfare_matching']
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"Wrote {len(csv_rows)} rows to {csv_file}")
PYEOF

echo "All experiments completed!"
