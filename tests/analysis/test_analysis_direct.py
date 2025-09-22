"""
Direct tests for analysis functionality without package imports.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

# Get project root for direct module loading
project_root = Path(__file__).parent.parent.parent


def load_module_from_path(module_name, file_path):
    """Load a module directly from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load our analysis modules directly
fisher_lda_path = project_root / "abstainer/src/analysis/fisher_lda.py"
direction_analysis_path = project_root / "abstainer/src/analysis/direction_analysis.py"

fisher_lda = load_module_from_path("fisher_lda", fisher_lda_path)
direction_analysis = load_module_from_path(
    "direction_analysis", direction_analysis_path
)


def test_compute_fisher_lda_direction():
    """Test Fisher LDA direction computation."""
    print("Testing Fisher LDA direction computation...")

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
    direction, metadata = fisher_lda.compute_fisher_lda_direction(H_pos, H_neg)

    # Verify output properties
    assert direction.shape == (n_features,)
    assert np.allclose(
        np.linalg.norm(direction), 1.0, rtol=1e-5
    )  # Should be normalized

    # Verify metadata
    assert metadata["n_pos"] == n_samples
    assert metadata["n_neg"] == n_samples
    assert "lambda" in metadata
    assert "alpha" in metadata

    print("  ✓ Direction shape and normalization correct")
    print("  ✓ Metadata contains expected fields")


def test_evaluate_direction():
    """Test direction evaluation."""
    print("Testing direction evaluation...")

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

    evaluation = direction_analysis.evaluate_direction(H_pos, H_neg, direction)

    # Should have good separation
    assert evaluation["auc"] > 0.9, f"AUC {evaluation['auc']} should be > 0.9"
    assert (
        evaluation["cohen_d"] > 2.0
    ), f"Cohen's d {evaluation['cohen_d']} should be > 2.0"
    assert evaluation["n_pos"] == n_samples
    assert evaluation["n_neg"] == n_samples

    # Check projection statistics
    assert (
        evaluation["mean_pos"] > evaluation["mean_neg"]
    ), "Positive class should project higher"

    print(f"  ✓ AUC: {evaluation['auc']:.4f} (> 0.9)")
    print(f"  ✓ Cohen's d: {evaluation['cohen_d']:.4f} (> 2.0)")
    print(f"  ✓ Mean separation: {evaluation['mean_pos'] - evaluation['mean_neg']:.4f}")


def test_integration():
    """Test Fisher LDA + evaluation integration."""
    print("Testing Fisher LDA + evaluation integration...")

    np.random.seed(42)

    n_samples = 200
    n_features = 30

    # Create data with subtle separation
    H_pos = np.random.randn(n_samples, n_features)
    H_pos[:, :5] += 1.0  # Slight shift in first 5 features

    H_neg = np.random.randn(n_samples, n_features)
    H_neg[:, :5] -= 1.0  # Opposite shift

    # Compute Fisher LDA direction
    direction, metadata = fisher_lda.compute_fisher_lda_direction(
        H_pos, H_neg, lambda_=-1.0
    )

    # Evaluate the computed direction
    evaluation = direction_analysis.evaluate_direction(H_pos, H_neg, direction)

    # Should achieve reasonable separation
    assert evaluation["auc"] > 0.7, f"AUC {evaluation['auc']} should be > 0.7"
    assert (
        evaluation["cohen_d"] > 0.5
    ), f"Cohen's d {evaluation['cohen_d']} should be > 0.5"

    print(f"  ✓ Integrated AUC: {evaluation['auc']:.4f}")
    print(f"  ✓ Integrated Cohen's d: {evaluation['cohen_d']:.4f}")
    print(f"  ✓ Used Ledoit-Wolf lambda: {metadata['lambda']:.4f}")


def main():
    """Run all tests."""
    print("Running direct analysis module tests...\n")

    try:
        test_compute_fisher_lda_direction()
        print()

        test_evaluate_direction()
        print()

        test_integration()
        print()

        print("All tests passed! ✅")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
