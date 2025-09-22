"""
Direction analysis and evaluation for abstention research.

This module provides functionality for:
- Evaluating the effectiveness of abstention directions
- Computing metrics like AUC, Cohen's d, and separation statistics
- Managing evaluation results across multiple layers
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

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
