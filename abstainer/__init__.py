# abstainer/__init__.py

"""
Abstainer package for analyzing abstention directions in language models.

Main API functions:
    - load_model: Load models and tokenizers
    - run_combined_experiment: Run experiments with Likert scoring and hidden states
    - get_questions_by_filter: Load and filter question datasets
    - build_prompt: Build prompts for different forms
    - decision_rule: Apply decision rules to model outputs
"""

# Core functionality
# Keep submodule access for advanced usage
from . import src

# Analysis and utilities
from .src.analysis import (
    DirectionAnalyzer,
    FisherLDAAnalyzer,
    ResidualVectorLoader,
    compute_fisher_lda_direction,
    evaluate_direction,
)
from .src.baselines import *
from .src.eval_utils import *
from .src.experiment import (
    load_combined_results,
    load_hidden_states,
    run_combined_experiment,
    run_hidden_states_experiment,
)
from .src.experiment_utils import get_questions_by_filter
from .src.io_utils import *
from .src.metrics import *
from .src.model import (
    load_model,
    next_token_logits,
    probs_from_logits,
    tokenizer_token_ids,
)
from .src.parsing import decision_rule
from .src.plots import *
from .src.probes import *
from .src.prompts import (
    build_likert_prompt,
    build_qualitative_prompt,
    build_quantitative_prompt,
    build_verbal_prompt,
)
from .src.steer import *

__version__ = "0.1.0"
__all__ = [
    # Core functions
    "load_model",
    "next_token_logits",
    "probs_from_logits",
    "tokenizer_token_ids",
    "run_combined_experiment",
    "run_hidden_states_experiment",
    "load_combined_results",
    "load_hidden_states",
    "get_questions_by_filter",
    "build_likert_prompt",
    "build_quantitative_prompt",
    "build_verbal_prompt",
    "build_qualitative_prompt",
    "decision_rule",
    # Submodule access
    "src",
]
