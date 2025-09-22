"""
Fisher Linear Discriminant Analysis for finding abstention directions.

This module implements Fisher LDA with Ledoit-Wolf shrinkage for robust
covariance estimation to find linear directions that separate abstention
from non-abstention examples in the residual stream.
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
from sklearn.covariance import ledoit_wolf

logger = logging.getLogger(__name__)


def compute_fisher_lda_direction(
    H_pos: np.ndarray, H_neg: np.ndarray, lambda_: float = 0.5, alpha: float = 1.0
) -> Tuple[np.ndarray, dict[str, Any]]:
    """
    Compute the Fisher LDA direction with shrinkage.

    Args:
        H_pos: Matrix of positive class examples (n_pos x d)
        H_neg: Matrix of negative class examples (n_neg x d)
        lambda_: Shrinkage coefficient. If negative, use Ledoit-Wolf estimation
        alpha: Identity scaling factor for shrinkage target

    Returns:
        Tuple of (normalized direction vector, metadata dict)

    Raises:
        ValueError: If input matrices have incompatible shapes
        np.linalg.LinAlgError: If covariance matrix is not invertible
    """
    if H_pos.shape[1] != H_neg.shape[1]:
        raise ValueError(
            f"Feature dimensions don't match: {H_pos.shape[1]} vs {H_neg.shape[1]}"
        )

    if H_pos.shape[0] == 0 or H_neg.shape[0] == 0:
        raise ValueError("Cannot compute LDA direction with empty class")

    # Compute means
    mu_pos = np.mean(H_pos, axis=0)
    mu_neg = np.mean(H_neg, axis=0)

    # Compute mean difference
    mean_diff = mu_pos - mu_neg

    # Compute pooled covariance
    n_pos = H_pos.shape[0]
    n_neg = H_neg.shape[0]
    n_total = n_pos + n_neg

    # Center the data
    H_pos_centered = H_pos - mu_pos
    H_neg_centered = H_neg - mu_neg

    # Compute individual covariances
    cov_pos = np.dot(H_pos_centered.T, H_pos_centered) / n_pos
    cov_neg = np.dot(H_neg_centered.T, H_neg_centered) / n_neg

    # Compute pooled covariance
    cov_pooled = ((n_pos * cov_pos) + (n_neg * cov_neg)) / n_total

    # Apply shrinkage
    d = cov_pooled.shape[0]  # Dimensionality

    # Use Ledoit-Wolf shrinkage if requested
    if lambda_ < 0:
        try:
            # Combine the data for Ledoit-Wolf estimation
            X_combined = np.vstack([H_pos, H_neg])

            # Estimate the optimal shrinkage using Ledoit-Wolf
            shrinkage_cov, lambda_estimated = ledoit_wolf(X_combined)
            lambda_ = lambda_estimated
            logger.info(f"Using Ledoit-Wolf estimated lambda: {lambda_:.4f}")
        except Exception as e:
            logger.warning(
                f"Ledoit-Wolf estimation failed: {e}. Using default lambda: 0.5"
            )
            lambda_ = 0.5

    # Apply the shrinkage
    shrinkage_target = alpha * np.eye(d)
    cov_shrunk = (1 - lambda_) * cov_pooled + lambda_ * shrinkage_target

    # Compute whitened mean difference
    try:
        # Try standard inverse first
        cov_inv = np.linalg.inv(cov_shrunk)
    except np.linalg.LinAlgError:
        # Fall back to pseudo-inverse if standard inverse fails
        logger.warning("Using pseudo-inverse for numerical stability")
        cov_inv = np.linalg.pinv(cov_shrunk)

    v = np.dot(cov_inv, mean_diff)

    # Normalize
    v_norm = np.linalg.norm(v)
    if v_norm > 0:
        v = v / v_norm
    else:
        logger.warning("Zero norm direction vector - classes may be identical")

    metadata = {
        "mu_pos": mu_pos,
        "mu_neg": mu_neg,
        "lambda": lambda_,
        "alpha": alpha,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "mean_diff_norm": np.linalg.norm(mean_diff),
        "shrinkage_target_trace": np.trace(shrinkage_target),
    }

    return v, metadata


class FisherLDAAnalyzer:
    """
    A class for managing Fisher LDA analysis across multiple layers.

    This class handles:
    - Computing LDA directions for multiple layers
    - Saving and loading results with checkpointing
    - Managing analysis metadata and provenance
    """

    def __init__(
        self,
        results_dir: Union[str, Path] = "./results/LDA",
        lambda_: float = -1.0,  # Use Ledoit-Wolf by default
        alpha: float = 1.0,
    ):
        """
        Initialize the Fisher LDA analyzer.

        Args:
            results_dir: Directory to save/load LDA results
            lambda_: Shrinkage coefficient (-1 for Ledoit-Wolf estimation)
            alpha: Identity scaling factor for shrinkage target
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.lambda_ = lambda_
        self.alpha = alpha

        self.lda_directions: dict[int, np.ndarray] = {}
        self.lda_metadata: dict[int, dict[str, Any]] = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_existing_results(self) -> bool:
        """
        Load existing LDA results if available.

        Returns:
            True if results were loaded, False otherwise
        """
        lda_files = list(self.results_dir.glob("lda_results_*.pkl"))

        if not lda_files:
            logger.info("No existing LDA results found")
            return False

        latest_lda_file = max(lda_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Loading existing LDA results from: {latest_lda_file}")

        try:
            with open(latest_lda_file, "rb") as f:
                existing_lda_data = pickle.load(f)

            if (
                "lda_directions" in existing_lda_data
                and "lda_metadata" in existing_lda_data
            ):
                self.lda_directions = existing_lda_data["lda_directions"].copy()
                self.lda_metadata = existing_lda_data["lda_metadata"].copy()
                logger.info(
                    f"Loaded {len(self.lda_directions)} existing layer directions"
                )
                return True
            else:
                logger.warning("Existing file doesn't contain expected data structure")
                return False

        except Exception as e:
            logger.error(f"Error loading existing LDA data: {e}")
            return False

    def compute_directions(
        self,
        residual_vectors: dict[str, dict[int, np.ndarray]],
        layers: Optional[list] = None,
        save_incremental: bool = True,
    ) -> dict[int, np.ndarray]:
        """
        Compute Fisher LDA directions for specified layers.

        Args:
            residual_vectors: Dict with 'positive' and 'negative' keys containing
                            layer-indexed arrays
            layers: List of layer indices to process (None for all available)
            save_incremental: Whether to save results after each layer

        Returns:
            Dictionary mapping layer indices to direction vectors
        """
        H_pos = residual_vectors["positive"]
        H_neg = residual_vectors["negative"]

        # Determine which layers to process
        if layers is None:
            layers_to_process = sorted(H_pos.keys())
        else:
            layers_to_process = sorted(layers)

        # Filter to only layers that need computation
        already_computed = set(self.lda_directions.keys())
        layers_to_compute = [
            layer
            for layer in layers_to_process
            if layer not in already_computed and layer in H_neg
        ]

        if not layers_to_compute:
            logger.info("All specified layers already have LDA directions computed")
            return self.lda_directions

        logger.info(
            f"Computing directions for {len(layers_to_compute)} layers: {layers_to_compute}"
        )

        for layer in layers_to_compute:
            logger.info(f"Computing direction for layer {layer}...")

            try:
                direction, metadata = compute_fisher_lda_direction(
                    H_pos[layer], H_neg[layer], lambda_=self.lambda_, alpha=self.alpha
                )

                self.lda_directions[layer] = direction
                self.lda_metadata[layer] = metadata

                logger.info(f"  Direction shape: {direction.shape}")
                logger.info(
                    f"  Positive examples: {metadata['n_pos']}, "
                    f"Negative examples: {metadata['n_neg']}"
                )

                if save_incremental:
                    self._save_incremental_results()

            except Exception as e:
                logger.error(f"Error computing direction for layer {layer}: {e}")

        logger.info(f"Total directions available: {len(self.lda_directions)} layers")

        # Save final complete results
        self._save_complete_results()

        return self.lda_directions

    def _save_incremental_results(self):
        """Save incremental results during computation."""
        save_path = self.results_dir / f"lda_results_{self.timestamp}_incremental.pkl"
        save_data = {
            "lda_directions": self.lda_directions,
            "lda_metadata": self.lda_metadata,
            "timestamp": self.timestamp,
            "lambda": self.lambda_,
            "alpha": self.alpha,
        }

        with open(save_path, "wb") as f:
            pickle.dump(save_data, f)

    def _save_complete_results(self):
        """Save final complete results."""
        save_path = self.results_dir / f"lda_results_{self.timestamp}_complete.pkl"
        save_data = {
            "lda_directions": self.lda_directions,
            "lda_metadata": self.lda_metadata,
            "timestamp": self.timestamp,
            "lambda": self.lambda_,
            "alpha": self.alpha,
            "total_layers": len(self.lda_directions),
        }

        with open(save_path, "wb") as f:
            pickle.dump(save_data, f)
        logger.info(f"Saved complete results to {save_path}")

    def get_directions(self) -> dict[int, np.ndarray]:
        """Get the computed LDA directions."""
        return self.lda_directions.copy()

    def get_metadata(self) -> dict[int, dict[str, Any]]:
        """Get the metadata for computed directions."""
        return self.lda_metadata.copy()
