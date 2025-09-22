# MATS Abstention Direction Project

## Scientific Overview

This project investigates **epistemic abstention in large language models** - whether LLMs encode "I don't know" as a simple internal feature that can be causally steered. We address the fundamental question: *Do models have a mechanistic one-dimensional knob for epistemic abstention?*

### Key Research Question
**Do LLMs encode epistemic abstention ("I don't know") as a simple internal feature that can be causally steered?**

### Dataset Creation & Curation
Existing benchmarks proved inadequate for studying epistemic abstention, as they focus on safety refusals or post-hoc calibration rather than genuine uncertainty, or suffer from severe compositional generalization issues. We constructed a Likert-scale dataset using **Gemini 2.5 Pro** (leveraging its large context window for comprehensive generation):

- **LLM-assisted generation**: Carefully crafted prompts to generate factual Yes/No questions across 14 subjects and 5 difficulty levels
- **Likert-style design**: Explicit abstention options alongside graded confidence to isolate epistemic uncertainty  
- **Blind curation**: LLM blind cross-checks to ensure question quality

**Dataset & Tools Location:**
- 📝 **System & user prompts**: [`abstainer/dataset/prompts/`](abstainer/dataset/prompts/) (generation and curation prompts)
- 🛠️ **Generation & quality control scripts**: [`abstainer/dataset/scripts/`](abstainer/dataset/scripts/) (including [`check_blind.py`](abstainer/dataset/scripts/check_blind.py) for dataset validation and comparison)
- 📊 **Curated dataset**: [`abstainer/dataset/data/curated_questions.tsv`](abstainer/dataset/data/curated_questions.tsv)

### Key Findings
Using this curated dataset, we discovered that **abstention is encoded as a single linear direction** in the residual stream:

- **Near-perfect separability**: AUC ≈ 0.99 on training data at layer 22 of `gemma-3-4b-it`
- **Strong generalization**: AUC = 0.79 on held-out data with large effect size (Cohen's d = 1.21)  
- **Late-layer concentration**: Separability emerges in mid-to-late layers, peaking around layer 22

### Research Methods
1. **Fisher Linear Discriminant Analysis** to identify abstention directions in representation space
2. **Mechanistic probing** of internal model states during abstention decisions
3. **Cross-validation** across prompt variants and held-out subjects
4. **Geometric analysis** of how models represent epistemic uncertainty

This work opens the door to **causal steering experiments** for improving model calibration and provides mechanistic insights into where epistemic uncertainty lives inside language models.

## Package Description

A Python package for analyzing abstention directions in language models using Fisher Linear Discriminant Analysis and other computational techniques.

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
