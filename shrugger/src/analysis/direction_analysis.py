"""
Direction analysis and evaluation for abstention research.

This module provides functionality for:
- Evaluating the effectiveness of abstention directions
- Computing metrics like AUC, Cohen's d, and separation statistics
- Managing evaluation results across multiple layers
- Plotting and saving evaluation visualizations
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# Import the save_figure function from plots module
try:
    from ..plots import save_figure
except ImportError:
    # When imported directly in tests
    import os
    import sys

    # Add the parent directory to sys.path
    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

    from plots import save_figure

logger = logging.getLogger(__name__)


def evaluate_direction(
    H_pos: np.ndarray, H_neg: np.ndarray, direction: np.ndarray
) -> dict[str, Any]:
    """
    Evaluate the effectiveness of a direction by projecting examples onto it.

    Args:
        H_pos: Matrix of positive class examples (n_pos x d)
        H_neg: Matrix of negative class examples (n_neg x d)
        direction: Direction vector to project onto (d,)

    Returns:
        Dictionary with evaluation metrics:
        - auc: ROC AUC score
        - cohen_d: Cohen's d effect size
        - mean_pos, mean_neg: Mean projections for each class
        - std_pos, std_neg: Standard deviations for each class
        - proj_pos, proj_neg: Raw projection values
        - separation: Mean difference between classes
        - overlap: Fraction of examples that would be misclassified at optimal threshold

    Raises:
        ValueError: If input shapes are incompatible or classes are empty
    """
    if H_pos.shape[1] != H_neg.shape[1]:
        raise ValueError(
            f"Feature dimensions don't match: {H_pos.shape[1]} vs {H_neg.shape[1]}"
        )

    if H_pos.shape[1] != len(direction):
        raise ValueError(
            f"Direction dimension {len(direction)} doesn't match features {H_pos.shape[1]}"
        )

    if H_pos.shape[0] == 0 or H_neg.shape[0] == 0:
        raise ValueError("Cannot evaluate direction with empty class")

    # Project examples onto the direction
    proj_pos = np.dot(H_pos, direction)
    proj_neg = np.dot(H_neg, direction)

    # Combine projections and labels for ROC analysis
    all_projections = np.concatenate([proj_pos, proj_neg])
    all_labels = np.concatenate([np.ones(len(proj_pos)), np.zeros(len(proj_neg))])

    # Compute ROC AUC
    try:
        auc = roc_auc_score(all_labels, all_projections)
    except ValueError as e:
        logger.warning(f"Could not compute AUC: {e}")
        auc = 0.5  # Random performance

    # Compute means and standard deviations
    mean_pos = np.mean(proj_pos)
    mean_neg = np.mean(proj_neg)
    std_pos = np.std(proj_pos)
    std_neg = np.std(proj_neg)

    # Compute Cohen's d (effect size)
    pooled_std = np.sqrt((std_pos**2 + std_neg**2) / 2)
    cohen_d = abs(mean_pos - mean_neg) / pooled_std if pooled_std > 0 else 0

    # Compute separation and overlap metrics
    separation = abs(mean_pos - mean_neg)

    # Estimate overlap by finding optimal threshold and computing error rate
    try:
        fpr, tpr, thresholds = roc_curve(all_labels, all_projections)
        # Find threshold that minimizes error rate (maximizes accuracy)
        error_rates = fpr * (len(proj_neg) / len(all_projections)) + (1 - tpr) * (
            len(proj_pos) / len(all_projections)
        )
        optimal_idx = np.argmin(error_rates)
        overlap = error_rates[optimal_idx]
    except Exception:
        overlap = 0.5  # Fallback to random performance

    return {
        "auc": auc,
        "mean_pos": mean_pos,
        "mean_neg": mean_neg,
        "std_pos": std_pos,
        "std_neg": std_neg,
        "cohen_d": cohen_d,
        "proj_pos": proj_pos,
        "proj_neg": proj_neg,
        "separation": separation,
        "overlap": overlap,
        "n_pos": len(proj_pos),
        "n_neg": len(proj_neg),
    }


class DirectionAnalyzer:
    """
    A class for analyzing and evaluating abstention directions across layers.

    This class handles:
    - Evaluating directions across multiple layers
    - Ranking layers by performance metrics
    - Generating summary statistics and reports
    """

    def __init__(self):
        """Initialize the direction analyzer."""
        self.evaluations: dict[int, dict[str, Any]] = {}
        self.summary_df: Optional[pd.DataFrame] = None

    def evaluate_all_layers(
        self,
        residual_vectors: dict[str, dict[int, np.ndarray]],
        lda_directions: dict[int, np.ndarray],
        layers: Optional[list[int]] = None,
    ) -> dict[int, dict[str, Any]]:
        """
        Evaluate directions for all specified layers.

        Args:
            residual_vectors: Dict with 'positive' and 'negative' keys
            lda_directions: Dict mapping layer indices to direction vectors
            layers: List of layers to evaluate (None for all available)

        Returns:
            Dictionary mapping layer indices to evaluation results
        """
        H_pos = residual_vectors["positive"]
        H_neg = residual_vectors["negative"]

        if layers is None:
            layers_to_evaluate = sorted(lda_directions.keys())
        else:
            layers_to_evaluate = sorted(layers)

        # Filter to only layers that have both data and directions
        available_layers = [
            layer
            for layer in layers_to_evaluate
            if layer in lda_directions and layer in H_pos and layer in H_neg
        ]

        logger.info(f"Evaluating directions for {len(available_layers)} layers...")

        self.evaluations = {}
        for layer in available_layers:
            logger.info(f"Evaluating direction for layer {layer}...")

            try:
                evaluation = evaluate_direction(
                    H_pos[layer], H_neg[layer], lda_directions[layer]
                )
                self.evaluations[layer] = evaluation

                logger.info(
                    f"  AUC: {evaluation['auc']:.4f}, "
                    f"Cohen's d: {evaluation['cohen_d']:.4f}"
                )

            except Exception as e:
                logger.error(f"Error evaluating direction for layer {layer}: {e}")

        if self.evaluations:
            self._create_summary_dataframe()
            best_layer = self.get_best_layer()
            logger.info(
                f"Best layer: {best_layer} with AUC {self.evaluations[best_layer]['auc']:.4f}"
            )

        return self.evaluations

    def _create_summary_dataframe(self):
        """Create a summary DataFrame from evaluation results."""
        if not self.evaluations:
            self.summary_df = pd.DataFrame()
            return

        summary_data = []
        for layer, eval_result in self.evaluations.items():
            summary_data.append(
                {
                    "layer": layer,
                    "auc": eval_result["auc"],
                    "cohen_d": eval_result["cohen_d"],
                    "mean_pos": eval_result["mean_pos"],
                    "mean_neg": eval_result["mean_neg"],
                    "std_pos": eval_result["std_pos"],
                    "std_neg": eval_result["std_neg"],
                    "separation": eval_result["separation"],
                    "overlap": eval_result["overlap"],
                    "n_pos": eval_result["n_pos"],
                    "n_neg": eval_result["n_neg"],
                }
            )

        self.summary_df = pd.DataFrame(summary_data)
        # Sort by AUC descending
        self.summary_df = self.summary_df.sort_values(
            "auc", ascending=False
        ).reset_index(drop=True)

    def get_best_layer(self, metric: str = "auc") -> Optional[int]:
        """
        Get the best layer based on a specified metric.

        Args:
            metric: Metric to use for ranking ('auc', 'cohen_d', 'separation')

        Returns:
            Layer index with best performance, or None if no evaluations
        """
        if not self.evaluations:
            return None

        if metric == "auc":
            return max(
                self.evaluations.keys(),
                key=lambda layer: self.evaluations[layer]["auc"],
            )
        elif metric == "cohen_d":
            return max(
                self.evaluations.keys(),
                key=lambda layer: self.evaluations[layer]["cohen_d"],
            )
        elif metric == "separation":
            return max(
                self.evaluations.keys(),
                key=lambda layer: self.evaluations[layer]["separation"],
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_top_layers(self, n: int = 10, metric: str = "auc") -> list[int]:
        """
        Get the top N layers based on a specified metric.

        Args:
            n: Number of top layers to return
            metric: Metric to use for ranking

        Returns:
            List of layer indices sorted by performance (best first)
        """
        if self.summary_df is None or len(self.summary_df) == 0:
            return []

        sorted_df = self.summary_df.sort_values(metric, ascending=False)
        return sorted_df.head(n)["layer"].tolist()

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Get the summary DataFrame with all evaluation results."""
        if self.summary_df is None:
            return pd.DataFrame()
        return self.summary_df.copy()

    def get_layer_evaluation(self, layer: int) -> Optional[dict[str, Any]]:
        """Get evaluation results for a specific layer."""
        return self.evaluations.get(layer)

    def print_summary(self, n_top: int = 10):
        """Print a summary of the top performing layers."""
        if self.summary_df is None or len(self.summary_df) == 0:
            print("No evaluation results available")
            return

        print(f"\nTop {min(n_top, len(self.summary_df))} layers by AUC:")
        print(
            self.summary_df.head(n_top)[
                ["layer", "auc", "cohen_d", "separation", "n_pos", "n_neg"]
            ].to_string(index=False)
        )

        best_layer = self.get_best_layer()
        if best_layer is not None:
            best_eval = self.evaluations[best_layer]
            print(f"\nBest layer: {best_layer}")
            print(f"  AUC: {best_eval['auc']:.4f}")
            print(f"  Cohen's d: {best_eval['cohen_d']:.4f}")
            print(f"  Separation: {best_eval['separation']:.4f}")
            print(f"  Overlap: {best_eval['overlap']:.4f}")

    def get_evaluations(self) -> dict[int, dict[str, Any]]:
        """Get all evaluation results."""
        return self.evaluations.copy()

    def plot_layer_metrics(
        self, metric="auc", figsize=(12, 6), title=None, save=True, filename=None
    ):
        """
        Plot a specific metric across all layers.

        Args:
            metric: Metric to plot ('auc', 'cohen_d', 'separation')
            figsize: Figure size as (width, height)
            title: Plot title (default: auto-generated based on metric)
            save: Whether to save the figure
            filename: Filename to use when saving (default: auto-generated)

        Returns:
            Matplotlib figure and axes
        """
        if not self.evaluations:
            logger.warning("No evaluations available to plot")
            return None, None

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Get data
        layers = sorted(self.evaluations.keys())
        values = [self.evaluations[layer][metric] for layer in layers]

        # Plot
        ax.plot(layers, values, "o-", linewidth=2, markersize=8)

        # Highlight best layer
        best_layer = self.get_best_layer(metric=metric)
        if best_layer is not None:
            best_value = self.evaluations[best_layer][metric]
            ax.scatter(
                [best_layer],
                [best_value],
                color="red",
                s=150,
                zorder=5,
                label=f"Best: Layer {best_layer}",
            )

        # Set labels and title
        metric_labels = {
            "auc": "AUC Score",
            "cohen_d": "Cohen's d Effect Size",
            "separation": "Class Separation",
            "overlap": "Class Overlap",
        }

        ax.set_xlabel("Layer", fontsize=14)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=14)

        if title is None:
            title = f"{metric_labels.get(metric, metric)} Across Layers"
        ax.set_title(title, fontsize=16)

        # Add grid and legend
        ax.grid(True, linestyle="--", alpha=0.7)
        if best_layer is not None:
            ax.legend()

        # Tight layout
        fig.tight_layout()

        # Save figure if requested
        if save:
            if filename is None:
                metric_name = metric.lower().replace(" ", "_")
                filename = f"layer_{metric_name}_comparison"
            save_figure(fig=fig, filename=filename)

        return fig, ax

    def plot_best_layer_projections(
        self, figsize=(12, 6), title=None, save=True, filename=None
    ):
        """
        Plot projections for the best layer.

        Args:
            figsize: Figure size as (width, height)
            title: Plot title (default: auto-generated)
            save: Whether to save the figure
            filename: Filename to use when saving (default: auto-generated)

        Returns:
            Matplotlib figure and axes
        """
        best_layer = self.get_best_layer()
        if best_layer is None:
            logger.warning("No best layer found")
            return None, None

        best_eval = self.evaluations[best_layer]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Get data
        proj_pos = best_eval["proj_pos"]
        proj_neg = best_eval["proj_neg"]
        mean_pos = best_eval["mean_pos"]
        mean_neg = best_eval["mean_neg"]

        # Plot histograms
        try:
            import seaborn as sns

            sns.histplot(
                proj_pos,
                bins=30,
                alpha=0.7,
                label="Positive Class",
                color="skyblue",
                ax=ax,
            )
            sns.histplot(
                proj_neg,
                bins=30,
                alpha=0.7,
                label="Negative Class",
                color="salmon",
                ax=ax,
            )
        except ImportError:
            ax.hist(
                proj_pos, bins=30, alpha=0.7, label="Positive Class", color="skyblue"
            )
            ax.hist(
                proj_neg, bins=30, alpha=0.7, label="Negative Class", color="salmon"
            )

        # Add vertical lines for means
        ax.axvline(
            mean_pos,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Mean Positive: {mean_pos:.4f}",
        )
        ax.axvline(
            mean_neg,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean Negative: {mean_neg:.4f}",
        )

        # Add metrics to the plot
        ax.text(
            0.02,
            0.95,
            f"AUC: {best_eval['auc']:.4f}",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.text(
            0.02,
            0.90,
            f"Cohen's d: {best_eval['cohen_d']:.4f}",
            transform=ax.transAxes,
            fontsize=12,
        )

        # Set title and labels
        if title is None:
            title = f"Layer {best_layer} - Projections onto Abstention Direction"
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Projection Value", fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)

        # Add legend and grid
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)

        # Tight layout
        fig.tight_layout()

        # Save figure if requested
        if save:
            if filename is None:
                filename = f"best_layer_{best_layer}_projections"
            save_figure(fig=fig, filename=filename)

        return fig, ax

    def plot_performance_summary(
        self, top_n=10, figsize=(14, 8), save=True, filename=None
    ):
        """
        Plot a summary of performance metrics for top layers.

        Args:
            top_n: Number of top layers to include
            figsize: Figure size as (width, height)
            save: Whether to save the figure
            filename: Filename to use when saving (default: auto-generated)

        Returns:
            Matplotlib figure and axes
        """
        if self.summary_df is None or len(self.summary_df) == 0:
            logger.warning("No summary data available")
            return None, None

        # Get top layers
        top_layers = self.summary_df.head(top_n)

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot AUC scores
        layers = top_layers["layer"].values
        aucs = top_layers["auc"].values
        ax1.bar(layers, aucs, color="skyblue")
        ax1.set_title("AUC Scores by Layer", fontsize=14)
        ax1.set_xlabel("Layer", fontsize=12)
        ax1.set_ylabel("AUC Score", fontsize=12)
        ax1.grid(axis="y", alpha=0.3)

        # Plot Cohen's d values
        cohens_d = top_layers["cohen_d"].values
        ax2.bar(layers, cohens_d, color="salmon")
        ax2.set_title("Cohen's d by Layer", fontsize=14)
        ax2.set_xlabel("Layer", fontsize=12)
        ax2.set_ylabel("Cohen's d", fontsize=12)
        ax2.grid(axis="y", alpha=0.3)

        # Add overall title
        fig.suptitle(f"Performance Metrics for Top {top_n} Layers", fontsize=16)

        # Tight layout
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle

        # Save figure if requested
        if save:
            if filename is None:
                filename = f"top_{top_n}_layers_performance"
            save_figure(fig=fig, filename=filename)

        return fig, (ax1, ax2)


def create_abstention_labels(metadata, abstention_answers=None):
    """
    Create abstention labels from question metadata.

    Args:
        metadata: List of dictionaries with question metadata
        abstention_answers: List of answer strings that indicate abstention
            (default: ['unanswerable', 'unknown', 'unclear', 'ambiguous'])

    Returns:
        Numpy array of binary labels (1 for abstention, 0 for non-abstention)
    """
    import numpy as np

    if abstention_answers is None:
        abstention_answers = ["unanswerable", "unknown", "unclear", "ambiguous"]

    # Create labels
    labels = []
    for meta in metadata:
        if meta["answer"].lower() in abstention_answers:
            labels.append(1)  # High confidence abstention
        else:
            labels.append(0)  # Low confidence abstention

    return np.array(labels)


def calculate_abstention_metrics(projections, labels):
    """
    Calculate metrics for abstention direction evaluation.

    Args:
        projections: Array of projection values
        labels: Binary labels (1 for abstention, 0 for non-abstention)

    Returns:
        Dictionary with evaluation metrics:
        - high_ca_projections: Projections for high CA examples
        - low_ca_projections: Projections for low CA examples
        - mean_high_ca: Mean projection for high CA examples
        - mean_low_ca: Mean projection for low CA examples
        - std_high_ca: Standard deviation for high CA examples
        - std_low_ca: Standard deviation for low CA examples
        - cohen_d: Cohen's d effect size
        - auc: ROC AUC score
        - all_projections: All projections concatenated
        - all_labels: All labels concatenated
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    # Convert labels to numpy array if not already
    labels = np.array(labels)

    # Split projections by label
    high_ca_projections = projections[labels == 1]
    low_ca_projections = projections[labels == 0]

    # Check if we have enough data
    if len(high_ca_projections) == 0 or len(low_ca_projections) == 0:
        return None

    # Calculate means and standard deviations
    mean_high_ca = np.mean(high_ca_projections)
    mean_low_ca = np.mean(low_ca_projections)
    std_high_ca = np.std(high_ca_projections)
    std_low_ca = np.std(low_ca_projections)

    # Calculate Cohen's d
    pooled_std = np.sqrt((std_high_ca**2 + std_low_ca**2) / 2)
    cohen_d = abs(mean_high_ca - mean_low_ca) / pooled_std if pooled_std > 0 else 0

    # Calculate AUC
    all_projections = np.concatenate([high_ca_projections, low_ca_projections])
    all_labels = np.concatenate(
        [np.ones(len(high_ca_projections)), np.zeros(len(low_ca_projections))]
    )
    auc = roc_auc_score(all_labels, all_projections)

    # Return metrics
    return {
        "high_ca_projections": high_ca_projections,
        "low_ca_projections": low_ca_projections,
        "mean_high_ca": mean_high_ca,
        "mean_low_ca": mean_low_ca,
        "std_high_ca": std_high_ca,
        "std_low_ca": std_low_ca,
        "cohen_d": cohen_d,
        "auc": auc,
        "all_projections": all_projections,
        "all_labels": all_labels,
        "n_high_ca": len(high_ca_projections),
        "n_low_ca": len(low_ca_projections),
    }


def plot_abstention_projections(
    metrics, title=None, figsize=(12, 6), save=True, filename=None
):
    """
    Plot histograms of abstention projections.

    Args:
        metrics: Dictionary of metrics from calculate_abstention_metrics
        title: Plot title (default: auto-generated)
        figsize: Figure size as (width, height)
        save: Whether to save the figure
        filename: Filename to use when saving (default: auto-generated)

    Returns:
        Matplotlib figure and axes
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get data
    high_ca_projections = metrics["high_ca_projections"]
    low_ca_projections = metrics["low_ca_projections"]
    mean_high_ca = metrics["mean_high_ca"]
    mean_low_ca = metrics["mean_low_ca"]

    # Plot histograms
    try:
        import seaborn as sns

        sns.histplot(
            high_ca_projections,
            bins=30,
            alpha=0.7,
            label="High CA (Abstention)",
            color="skyblue",
            ax=ax,
        )
        sns.histplot(
            low_ca_projections,
            bins=30,
            alpha=0.7,
            label="Low CA (Non-abstention)",
            color="salmon",
            ax=ax,
        )
    except ImportError:
        ax.hist(
            high_ca_projections,
            bins=30,
            alpha=0.7,
            label="High CA (Abstention)",
            color="skyblue",
        )
        ax.hist(
            low_ca_projections,
            bins=30,
            alpha=0.7,
            label="Low CA (Non-abstention)",
            color="salmon",
        )

    # Add vertical lines for means
    ax.axvline(
        mean_high_ca,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Mean High CA: {mean_high_ca:.4f}",
    )
    ax.axvline(
        mean_low_ca,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean Low CA: {mean_low_ca:.4f}",
    )

    # Add metrics to the plot
    ax.text(
        0.02, 0.95, f"AUC: {metrics['auc']:.4f}", transform=ax.transAxes, fontsize=12
    )
    ax.text(
        0.02,
        0.90,
        f"Cohen's d: {metrics['cohen_d']:.4f}",
        transform=ax.transAxes,
        fontsize=12,
    )

    # Set title and labels
    if title is None:
        title = "Projections onto Abstention Direction"
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Projection Value", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)

    # Add legend and grid
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    # Tight layout
    fig.tight_layout()

    # Save figure if requested
    if save:
        if filename is None:
            clean_title = (
                title.lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(":", "")
            )
            filename = f"abstention_projections_{clean_title}"
        save_figure(fig=fig, filename=filename)

    return fig, ax


def plot_abstention_roc_curve(
    metrics, title=None, figsize=(8, 8), save=True, filename=None
):
    """
    Plot ROC curve for abstention prediction.

    Args:
        metrics: Dictionary of metrics from calculate_abstention_metrics
        title: Plot title (default: auto-generated)
        figsize: Figure size as (width, height)
        save: Whether to save the figure
        filename: Filename to use when saving (default: auto-generated)

    Returns:
        Matplotlib figure and axes
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get data
    y_true = metrics["all_labels"]
    y_score = metrics["all_projections"]

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = metrics["auc"]

    # Plot ROC curve
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")

    # Set labels and title
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)

    if title is None:
        title = "ROC Curve for Abstention Prediction"
    ax.set_title(title, fontsize=14)

    # Add legend and grid
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Tight layout
    fig.tight_layout()

    # Save figure if requested
    if save:
        if filename is None:
            clean_title = (
                title.lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(":", "")
            )
            filename = f"abstention_roc_curve_{clean_title}"
        save_figure(fig=fig, filename=filename)

    return fig, ax
