"""
Standalone tests for analysis functionality that don't require full project imports.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

from abstainer import (
    DirectionAnalyzer,
    FisherLDAAnalyzer,
    compute_fisher_lda_direction,
    evaluate_direction,
)


def test_compute_fisher_lda_direction_basic():
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


def test_evaluate_direction_basic():
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


def test_fisher_lda_analyzer_initialization():
    """Test analyzer initialization."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        analyzer = FisherLDAAnalyzer(results_dir=tmp_dir)

        assert analyzer.results_dir == Path(tmp_dir)
        assert analyzer.lambda_ == -1.0  # Default Ledoit-Wolf
        assert analyzer.alpha == 1.0
        assert len(analyzer.lda_directions) == 0
        assert len(analyzer.lda_metadata) == 0


def test_direction_analyzer_basic():
    """Test the direction analyzer."""
    analyzer = DirectionAnalyzer()

    # Should start empty
    assert len(analyzer.evaluations) == 0
    assert analyzer.get_best_layer() is None


if __name__ == "__main__":
    # Run tests manually
    print("Running standalone analysis tests...")

    try:
        test_compute_fisher_lda_direction_basic()
        print("✓ Fisher LDA direction computation test passed")

        test_evaluate_direction_basic()
        print("✓ Direction evaluation test passed")

        test_fisher_lda_analyzer_initialization()
        print("✓ Fisher LDA analyzer initialization test passed")

        test_direction_analyzer_basic()
        print("✓ Direction analyzer basic test passed")

        print("\nAll tests passed! ✓")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
