"""
Analysis modules for abstention direction research.

This package contains modules for:
- Fisher LDA analysis for finding abstention directions
- Direction computation and evaluation
- Data loading and result management
- Metrics analysis and visualization
"""

from .data_loader import ResidualVectorLoader, load_dev_data, load_dev_form_data
from .direction_analysis import (
    DirectionAnalyzer,
    calculate_abstention_metrics,
    create_abstention_labels,
    evaluate_direction,
    plot_abstention_projections,
    plot_abstention_roc_curve,
)
from .fisher_lda import FisherLDAAnalyzer, compute_fisher_lda_direction
from .metrics_analysis import (
    MetricsAnalyzer,
    find_top_examples,
    load_metrics_data,
    plot_metric_distributions,
    plot_metrics_by_form,
    plot_metrics_by_label_type,
)

__all__ = [
    "FisherLDAAnalyzer",
    "compute_fisher_lda_direction",
    "DirectionAnalyzer",
    "evaluate_direction",
    "ResidualVectorLoader",
    "calculate_abstention_metrics",
    "create_abstention_labels",
    "load_dev_form_data",
    "load_dev_data",
    "plot_abstention_projections",
    "plot_abstention_roc_curve",
    "MetricsAnalyzer",
    "load_metrics_data",
    "plot_metrics_by_form",
    "plot_metrics_by_label_type",
    "plot_metric_distributions",
    "find_top_examples",
]
