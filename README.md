# MATS Abstention Direction Project

A Python package for analyzing abstention directions in language models using Fisher Linear Discriminant Analysis.

## Project Structure

```
├── abstainer/              # 📦 Main package source code
│   ├── src/               # Core modules (importable)
│   │   ├── analysis/      # Fisher LDA and direction analysis
│   │   ├── experiment.py  # Experiment running utilities
│   │   ├── model.py       # Model loading and inference
│   │   └── ...           # Other core modules
│   └── dataset/          # Dataset utilities
├── experiments/          # 🧪 Main experiment scripts
│   ├── run_comprehensive_experiments.py  # Full experiment suite
│   ├── run_metrics_analysis.py          # Results analysis
│   └── README.md         # Experiments documentation
├── examples/             # 🎯 Example scripts and demos
├── notebooks/            # 📓 Jupyter notebooks
│   ├── analysis/         # Analysis notebooks
│   └── exploration/      # Exploratory/sandbox notebooks
├── scripts/              # 🔧 Utility scripts
│   ├── analysis/         # Analysis and visualization scripts
│   ├── run_fisher_analysis.py  # Main analysis CLI
│   └── cleanup_corrupted_files.py  # Maintenance utilities
├── outputs/              # 📊 Generated outputs (gitignored)
│   ├── figures/          # Generated plots and visualizations
│   └── data/             # Processed/intermediate data files
├── results/              # 🗂️ Raw experiment results (gitignored)
│   ├── LDA/              # Fisher LDA analysis results
│   ├── comprehensive_experiments/  # Full experiment runs
│   └── ...               # Other experiment outputs
├── tests/                # ✅ Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── analysis/         # Analysis-specific tests
├── config.json.template  # 📝 Configuration template
├── CONFIG.md            # 📋 Configuration guide
├── environment.yml       # 🐍 Conda environment specification
├── pyproject.toml        # ⚙️ Package configuration
└── Makefile             # 🏗️ Build and test automation
```

### Directory Guidelines

- **`abstainer/`**: Package source code only - no outputs or experiments
- **`experiments/`**: Main experiment scripts for running comprehensive studies
- **`results/`**: Raw experiment data, model outputs, large files
- **`outputs/`**: Processed visualizations and analysis outputs  
- **`examples/`**: Demonstration scripts and sample usage
- **`notebooks/`**: Interactive analysis and exploration
- **`scripts/`**: Command-line utilities and batch processing

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
- **Professional Package Structure**: Clean, maintainable codebase following Python best practices

## Development

The package is installed in editable mode automatically when you create the environment, so any changes to the source code are immediately available.

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

## Directory Guidelines

- **Source code**: Only in `abstainer/` package
- **Experiments**: Main experiment scripts in `experiments/`
- **Notebooks**: Use `notebooks/analysis/` for analysis, `notebooks/exploration/` for experimentation
- **Scripts**: Utility scripts go in `scripts/`, with analysis scripts in `scripts/analysis/`
- **Outputs**: All generated files (plots, data) go in `outputs/`
- **Configuration**: Use `config.json` (from template) for tokens and settings
- **No clutter**: Keep project root clean - use appropriate subdirectories

## License

MIT License - see LICENSE file for details.
