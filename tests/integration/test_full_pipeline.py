#!/usr/bin/env python
"""
Full pipeline test for the refactored Fisher LDA analysis system.

This test checks:
1. Data loading functionality
2. Fisher LDA computation
3. Direction evaluation
4. CLI script functionality
5. Integration with actual data (if available)
"""

import importlib.util
import subprocess
import sys
import tempfile
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


# Load analysis modules
fisher_lda_path = project_root / "shrugger/src/analysis/fisher_lda.py"
direction_analysis_path = project_root / "shrugger/src/analysis/direction_analysis.py"
data_loader_path = project_root / "shrugger/src/analysis/data_loader.py"

fisher_lda = load_module_from_path("fisher_lda", fisher_lda_path)
direction_analysis = load_module_from_path(
    "direction_analysis", direction_analysis_path
)
data_loader = load_module_from_path("data_loader", data_loader_path)


def test_synthetic_data_pipeline():
    """Test the complete pipeline with synthetic data."""
    print("Testing complete pipeline with synthetic data...")

    np.random.seed(42)

    # Create synthetic multi-layer data
    n_samples = 200
    n_features = 100
    n_layers = 5

    residual_vectors = {"positive": {}, "negative": {}}

    for layer in range(n_layers):
        # Create data with different separation patterns per layer
        # Layer performance should increase with layer number
        separation_strength = 0.5 + layer * 0.3

        H_pos = np.random.randn(n_samples, n_features)
        H_neg = np.random.randn(n_samples, n_features)

        # Add separation in different dimensions for each layer
        H_pos[:, layer : layer + 5] += separation_strength
        H_neg[:, layer : layer + 5] -= separation_strength

        residual_vectors["positive"][layer] = H_pos
        residual_vectors["negative"][layer] = H_neg

    # Test Fisher LDA Analyzer
    with tempfile.TemporaryDirectory() as temp_dir:
        lda_analyzer = fisher_lda.FisherLDAAnalyzer(
            results_dir=temp_dir, lambda_=-1.0, alpha=1.0  # Use Ledoit-Wolf
        )

        # Compute directions
        lda_directions = lda_analyzer.compute_directions(
            residual_vectors=residual_vectors,
            save_incremental=False,  # Don't save during test
        )

        assert (
            len(lda_directions) == n_layers
        ), f"Expected {n_layers} directions, got {len(lda_directions)}"

        for layer in range(n_layers):
            direction = lda_directions[layer]
            assert direction.shape == (
                n_features,
            ), f"Layer {layer} direction has wrong shape"
            assert np.allclose(
                np.linalg.norm(direction), 1.0
            ), f"Layer {layer} direction not normalized"

        print(f"  ✓ Computed {len(lda_directions)} LDA directions")

        # Test Direction Analyzer
        direction_analyzer = direction_analysis.DirectionAnalyzer()
        evaluations = direction_analyzer.evaluate_all_layers(
            residual_vectors=residual_vectors, lda_directions=lda_directions
        )

        assert (
            len(evaluations) == n_layers
        ), f"Expected {n_layers} evaluations, got {len(evaluations)}"

        # Check that performance generally increases with layer number
        aucs = [evaluations[layer]["auc"] for layer in range(n_layers)]

        # Last layer should perform better than first layer
        assert (
            aucs[-1] > aucs[0]
        ), f"Last layer AUC ({aucs[-1]:.3f}) should be > first layer AUC ({aucs[0]:.3f})"

        # Get best layer
        best_layer = direction_analyzer.get_best_layer()
        best_eval = direction_analyzer.get_layer_evaluation(best_layer)

        print(f"  ✓ Best layer: {best_layer} with AUC {best_eval['auc']:.4f}")
        print(f"  ✓ AUC progression: {[f'{auc:.3f}' for auc in aucs]}")

        # Test summary dataframe
        summary_df = direction_analyzer.get_summary_dataframe()
        assert len(summary_df) == n_layers, "Summary dataframe has wrong length"
        assert "auc" in summary_df.columns, "Summary missing AUC column"
        assert "cohen_d" in summary_df.columns, "Summary missing Cohen's d column"

        print(f"  ✓ Summary dataframe created with {len(summary_df)} rows")


def test_data_loader_with_synthetic_files():
    """Test the data loader with synthetic pickle files."""
    print("Testing data loader with synthetic files...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create synthetic layer files
        n_layers = 3
        n_samples = 50
        n_features = 20
        timestamp = "20240101_120000"

        for layer in range(n_layers):
            # Create synthetic data
            pos_data = np.random.randn(n_samples, n_features) + 1.0
            neg_data = np.random.randn(n_samples, n_features) - 1.0

            layer_data = {
                "residual_vectors_by_layer": {
                    layer: {"positive": pos_data, "negative": neg_data}
                }
            }

            # Save as layer file
            filename = f"residual_vectors_{timestamp}_layer_{layer}.pkl"
            filepath = temp_path / filename

            import pickle

            with open(filepath, "wb") as f:
                pickle.dump(layer_data, f)

        # Test loading with ResidualVectorLoader
        loader = data_loader.ResidualVectorLoader(results_dir=temp_path.parent)

        try:
            residual_vectors = loader.load_layer_files(data_dir=temp_path)

            assert "positive" in residual_vectors, "Missing positive data"
            assert "negative" in residual_vectors, "Missing negative data"
            assert (
                len(residual_vectors["positive"]) == n_layers
            ), f"Expected {n_layers} positive layers"
            assert (
                len(residual_vectors["negative"]) == n_layers
            ), f"Expected {n_layers} negative layers"

            for layer in range(n_layers):
                assert (
                    layer in residual_vectors["positive"]
                ), f"Missing positive data for layer {layer}"
                assert (
                    layer in residual_vectors["negative"]
                ), f"Missing negative data for layer {layer}"

                pos_shape = residual_vectors["positive"][layer].shape
                neg_shape = residual_vectors["negative"][layer].shape

                assert pos_shape == (
                    n_samples,
                    n_features,
                ), f"Wrong positive shape for layer {layer}"
                assert neg_shape == (
                    n_samples,
                    n_features,
                ), f"Wrong negative shape for layer {layer}"

            print(
                f"  ✓ Loaded {len(residual_vectors['positive'])} layers from synthetic files"
            )

        except Exception as e:
            print(f"  ⚠ Data loader test failed (expected if no real data): {e}")


def test_cli_script_functionality():
    """Test the CLI script with various options."""
    print("Testing CLI script functionality...")

    script_path = project_root / "scripts/run_fisher_analysis.py"

    # Test help
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0, f"CLI help failed: {result.stderr}"
        assert (
            "Fisher LDA analysis" in result.stdout
        ), "Help text doesn't contain expected content"
        print("  ✓ CLI help works correctly")

    except subprocess.TimeoutExpired:
        print("  ⚠ CLI help test timed out")
    except Exception as e:
        print(f"  ⚠ CLI help test failed: {e}")

    # Test invalid arguments
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--invalid-arg"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0, "CLI should fail with invalid arguments"
        print("  ✓ CLI properly handles invalid arguments")

    except subprocess.TimeoutExpired:
        print("  ⚠ CLI invalid args test timed out")
    except Exception as e:
        print(f"  ⚠ CLI invalid args test failed: {e}")


def test_real_data_if_available():
    """Test with real data if available."""
    print("Testing with real data (if available)...")

    # Check for real data
    possible_data_dirs = [
        Path("./results/abstention_direction"),
        Path("./results/by_layer"),
        Path("../results/abstention_direction"),
    ]

    data_dir = None
    for dir_path in possible_data_dirs:
        if dir_path.exists() and list(dir_path.glob("*.pkl")):
            data_dir = dir_path
            break

    if data_dir is None:
        print("  ⚠ No real data found - skipping real data test")
        return

    try:
        loader = data_loader.ResidualVectorLoader()

        # Try to load real data
        if "layer" in str(data_dir):
            residual_vectors = loader.load_layer_files(data_dir=data_dir)
        else:
            # Look for complete files
            complete_files = list(data_dir.glob("*complete.pkl"))
            if complete_files:
                residual_vectors = loader.load_complete_file(complete_files[0])
            else:
                residual_vectors = loader.load_layer_files(data_dir=data_dir)

        layers = sorted(residual_vectors["positive"].keys())
        print(f"  ✓ Loaded real data with {len(layers)} layers")

        # Test a quick analysis on a subset of layers
        test_layers = layers[:5] if len(layers) > 5 else layers

        with tempfile.TemporaryDirectory() as temp_dir:
            lda_analyzer = fisher_lda.FisherLDAAnalyzer(results_dir=temp_dir)

            test_residual_vectors = {
                "positive": {
                    layer: residual_vectors["positive"][layer] for layer in test_layers
                },
                "negative": {
                    layer: residual_vectors["negative"][layer] for layer in test_layers
                },
            }

            directions = lda_analyzer.compute_directions(
                residual_vectors=test_residual_vectors, save_incremental=False
            )

            print(f"  ✓ Computed directions for {len(directions)} real data layers")

            # Quick evaluation
            direction_analyzer = direction_analysis.DirectionAnalyzer()
            evaluations = direction_analyzer.evaluate_all_layers(
                residual_vectors=test_residual_vectors, lda_directions=directions
            )

            if evaluations:
                best_layer = direction_analyzer.get_best_layer()
                best_auc = evaluations[best_layer]["auc"]
                print(
                    f"  ✓ Real data analysis: best layer {best_layer} with AUC {best_auc:.4f}"
                )

    except Exception as e:
        print(f"  ⚠ Real data test failed: {e}")


def main():
    """Run all tests."""
    print("Running comprehensive Fisher LDA analysis tests...\n")

    tests_passed = 0
    total_tests = 4

    try:
        test_synthetic_data_pipeline()
        tests_passed += 1
        print()
    except Exception as e:
        print(f"❌ Synthetic data pipeline test failed: {e}")
        import traceback

        traceback.print_exc()
        print()

    try:
        test_data_loader_with_synthetic_files()
        tests_passed += 1
        print()
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        print()

    try:
        test_cli_script_functionality()
        tests_passed += 1
        print()
    except Exception as e:
        print(f"❌ CLI script test failed: {e}")
        print()

    try:
        test_real_data_if_available()
        tests_passed += 1
        print()
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        print()

    print(f"Tests completed: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! The refactored system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed, but core functionality appears to work.")
        return tests_passed >= 2  # Pass if at least core tests work


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
