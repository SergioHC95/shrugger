"""
Tests for Fisher LDA analysis functionality.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from shrugger import (
    DirectionAnalyzer,
    FisherLDAAnalyzer,
    compute_fisher_lda_direction,
    evaluate_direction,
)


class TestFisherLDA:
    """Test cases for Fisher LDA functionality."""

    def test_compute_fisher_lda_direction_basic(self):
        """Test basic Fisher LDA direction computation."""
        np.random.seed(42)

        # Create synthetic data with clear separation
        n_samples = 100
        n_features = 50

        # Positive class: higher values in first half of features
        H_pos = np.random.randn(n_samples, n_features)
        H_pos[:, : n_features // 2] += 2.0

        # Negative class: higher values in second half of features
        H_neg = np.random.randn(n_samples, n_features)
        H_neg[:, n_features // 2 :] += 2.0

        # Compute direction
        direction, metadata = compute_fisher_lda_direction(H_pos, H_neg)

        # Verify output properties
        assert direction.shape == (n_features,)
        assert np.allclose(np.linalg.norm(direction), 1.0)  # Should be normalized

        # Verify metadata
        assert metadata["n_pos"] == n_samples
        assert metadata["n_neg"] == n_samples
        assert "lambda" in metadata
        assert "alpha" in metadata

    def test_compute_fisher_lda_direction_ledoit_wolf(self):
        """Test Fisher LDA with Ledoit-Wolf shrinkage."""
        np.random.seed(42)

        n_samples = 50
        n_features = 100  # High-dimensional case

        H_pos = np.random.randn(n_samples, n_features) + 1.0
        H_neg = np.random.randn(n_samples, n_features) - 1.0

        # Use Ledoit-Wolf estimation (lambda_ < 0)
        direction, metadata = compute_fisher_lda_direction(H_pos, H_neg, lambda_=-1.0)

        assert direction.shape == (n_features,)
        assert np.allclose(np.linalg.norm(direction), 1.0)
        assert 0 <= metadata["lambda"] <= 1  # Ledoit-Wolf should give valid shrinkage

    def test_compute_fisher_lda_direction_errors(self):
        """Test error handling in Fisher LDA computation."""
        # Mismatched dimensions
        H_pos = np.random.randn(10, 5)
        H_neg = np.random.randn(10, 3)

        with pytest.raises(ValueError, match="Feature dimensions don't match"):
            compute_fisher_lda_direction(H_pos, H_neg)

        # Empty classes
        H_pos_empty = np.random.randn(0, 5)
        H_neg_valid = np.random.randn(10, 5)

        with pytest.raises(
            ValueError, match="Cannot compute LDA direction with empty class"
        ):
            compute_fisher_lda_direction(H_pos_empty, H_neg_valid)


class TestDirectionAnalysis:
    """Test cases for direction analysis functionality."""

    def test_evaluate_direction_basic(self):
        """Test basic direction evaluation."""
        np.random.seed(42)

        n_samples = 100
        n_features = 20

        # Create data with known separation along first feature
        H_pos = np.random.randn(n_samples, n_features)
        H_pos[:, 0] += 3.0  # Shift positive class

        H_neg = np.random.randn(n_samples, n_features)
        H_neg[:, 0] -= 3.0  # Shift negative class

        # Direction that should separate well
        direction = np.zeros(n_features)
        direction[0] = 1.0  # Point along the separating dimension

        evaluation = evaluate_direction(H_pos, H_neg, direction)

        # Should have good separation
        assert evaluation["auc"] > 0.9  # Should be high AUC
        assert evaluation["cohen_d"] > 2.0  # Should have large effect size
        assert evaluation["n_pos"] == n_samples
        assert evaluation["n_neg"] == n_samples

        # Check projection statistics
        assert (
            evaluation["mean_pos"] > evaluation["mean_neg"]
        )  # Positive should project higher

    def test_evaluate_direction_errors(self):
        """Test error handling in direction evaluation."""
        # Mismatched dimensions
        H_pos = np.random.randn(10, 5)
        H_neg = np.random.randn(10, 5)
        direction = np.random.randn(3)  # Wrong dimension

        with pytest.raises(ValueError, match="Direction dimension"):
            evaluate_direction(H_pos, H_neg, direction)


class TestFisherLDAAnalyzer:
    """Test cases for the FisherLDAAnalyzer class."""

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            analyzer = FisherLDAAnalyzer(results_dir=tmp_dir)

            assert analyzer.results_dir == Path(tmp_dir)
            assert analyzer.lambda_ == -1.0  # Default Ledoit-Wolf
            assert analyzer.alpha == 1.0
            assert len(analyzer.lda_directions) == 0
            assert len(analyzer.lda_metadata) == 0

    def test_analyzer_compute_directions(self):
        """Test computing directions with the analyzer."""
        np.random.seed(42)

        with tempfile.TemporaryDirectory() as tmp_dir:
            analyzer = FisherLDAAnalyzer(results_dir=tmp_dir)

            # Create synthetic multi-layer data
            n_samples = 50
            n_features = 20
            layers = [0, 1, 2]

            residual_vectors = {"positive": {}, "negative": {}}

            for layer in layers:
                # Each layer has different separation patterns
                H_pos = np.random.randn(n_samples, n_features)
                H_pos[:, layer] += 2.0  # Different feature for each layer

                H_neg = np.random.randn(n_samples, n_features)
                H_neg[:, layer] -= 2.0

                residual_vectors["positive"][layer] = H_pos
                residual_vectors["negative"][layer] = H_neg

            # Compute directions
            directions = analyzer.compute_directions(
                residual_vectors,
                layers=layers,
                save_incremental=False,  # Don't save during test
            )

            assert len(directions) == len(layers)
            for layer in layers:
                assert layer in directions
                assert directions[layer].shape == (n_features,)
                assert np.allclose(np.linalg.norm(directions[layer]), 1.0)


class TestDirectionAnalyzer:
    """Test cases for the DirectionAnalyzer class."""

    def test_analyzer_evaluation(self):
        """Test the direction analyzer evaluation."""
        np.random.seed(42)

        analyzer = DirectionAnalyzer()

        # Create synthetic data
        n_samples = 100
        n_features = 10
        layers = [0, 1]

        residual_vectors = {"positive": {}, "negative": {}}

        lda_directions = {}

        for layer in layers:
            # Create data with separation along different dimensions
            H_pos = np.random.randn(n_samples, n_features)
            H_pos[:, layer] += 2.0

            H_neg = np.random.randn(n_samples, n_features)
            H_neg[:, layer] -= 2.0

            residual_vectors["positive"][layer] = H_pos
            residual_vectors["negative"][layer] = H_neg

            # Create direction along separating dimension
            direction = np.zeros(n_features)
            direction[layer] = 1.0
            lda_directions[layer] = direction

        # Evaluate all layers
        evaluations = analyzer.evaluate_all_layers(
            residual_vectors=residual_vectors,
            lda_directions=lda_directions,
            layers=layers,
        )

        assert len(evaluations) == len(layers)
        for layer in layers:
            assert layer in evaluations
            assert evaluations[layer]["auc"] > 0.8  # Should have good separation

        # Test best layer selection
        best_layer = analyzer.get_best_layer()
        assert best_layer in layers

        # Test summary dataframe
        summary_df = analyzer.get_summary_dataframe()
        assert len(summary_df) == len(layers)
        assert "auc" in summary_df.columns
        assert "cohen_d" in summary_df.columns


if __name__ == "__main__":
    pytest.main([__file__])
