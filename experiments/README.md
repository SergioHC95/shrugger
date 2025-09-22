# Experiments

This directory contains the main experiment scripts for the shrugger project.

## Scripts

### `run_comprehensive_experiments.py`
Runs comprehensive experiments across all prompt forms (V0-V5) and label permutations.

**Usage:**
```bash
python experiments/run_comprehensive_experiments.py
```

**Features:**
- Works in both local and Google Colab environments
- Automatically detects environment and sets up accordingly
- Runs 60 experiments total (6 forms × 5 permutations × 2 label types)
- Saves detailed logs and summaries

### `run_metrics_analysis.py`
Analyzes experiment results and computes metrics.

**Usage:**
```bash
python experiments/run_metrics_analysis.py
```

## Requirements

Make sure you have:
1. Set up the conda environment: `conda env create -f environment.yml && conda activate shrugger`
2. Created a `config.json` file (see `CONFIG.md` in project root)
3. Sufficient disk space for experiment results

## Output

Experiments save results to `results/comprehensive_experiments/` with timestamped directories for each run.
