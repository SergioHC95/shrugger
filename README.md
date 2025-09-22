# Mechanistic Steering for Hallucination Suppression

## Executive Summary
See [`REPORT.md`](REPORT.md)

## Overview

> **Do LLMs encode epistemic abstention as an internal feature that can be causally steered?**

This project investigates **epistemic abstention** in LLMs, i.e., whether models encode "I don't know" as a simple internal feature that can be mechanistically controlled to:
  * Reduce hallucinations in cases of epistemic uncertainty
  * Improve calibration metrics (ECE, AURC)
  * Control risk in high-stakes situations (cf. deontic refusal)


## Package Description

A Python package for analyzing abstention directions in language models using Fisher Linear Discriminant Analysis and other computational techniques.

## Project Structure

```
├── abstainer/             # 📦 Core package
│   ├── src/                  # Analysis, experiments, models
│   └── dataset/              # Data, prompts, curation scripts
├── examples/              # 🎯 Usage examples  
├── experiments/           # 🧪 Main experiment runners
├── notebooks/             # 📓 Analysis & exploration
├── scripts/               # 🛠️ Utility scripts
├── tests/                 # ✅ Test suite
├── results/               # 🗂️ Raw experiment results (large files, gitignored)
├── outputs/               # 📊 Processed data and visualizations (gitignored)
└── config files           # ⚙️ Environment & package setup
```


## Quick Start

1. **Create the environment:**
   ```bash
   conda env create -f environment.yml
   conda activate abstainer
   ```

2. **Set up configuration:**
   ```bash
   cp config.json.template config.json
   # Edit config.json with your Hugging Face token
   ```
   See `CONFIG.md` for detailed configuration instructions.

3. **Run tests:**
   ```bash
   make test
   ```

4. **Run experiments:**
   ```bash
   python experiments/run_comprehensive_experiments.py
   ```

5. **Run analysis:**
   ```bash
   python scripts/run_fisher_analysis.py --help
   ```

## Key Features

- **Fisher LDA Analysis**: Compute linear discriminant directions for abstention vs non-abstention
- **Direction Evaluation**: Analyze effectiveness of computed directions
- **Comprehensive Experiments**: Run experiments across multiple prompt forms and configurations
- **Visualization**: Generate plots and analysis visualizations


## Development

The package is installed in editable mode automatically when you create the environment.

### Running Different Test Suites

- All tests: `make test`
- Unit tests only: `make test-unit`
- Integration tests: `make test-integration`
- Analysis tests: `make test-analysis`

### Code Quality

- Linting: `make lint`
- Formatting: `make fix`
- Coverage: `make coverage`

## Main Scripts

- `experiments/run_comprehensive_experiments.py`: Run comprehensive experiments across configurations (works in Colab and local)
- `experiments/run_metrics_analysis.py`: Analyze experiment results and compute metrics  
- `scripts/run_fisher_analysis.py`: Main Fisher LDA analysis CLI
- `scripts/reorganize_by_layer.py`: Reorganize experiment data by layer
- `scripts/cleanup_corrupted_files.py`: Clean up corrupted NPZ files

## License

MIT License - see LICENSE file for details.
