#!/usr/bin/env python
"""
CLI script for running Fisher LDA analysis on abstention data.

This script provides a command-line interface for:
- Loading residual vectors from various sources
- Computing Fisher LDA directions across layers
- Evaluating direction effectiveness
- Generating summary reports and visualizations
"""

import argparse
import logging
import sys
from pathlib import Path

from shrugger import DirectionAnalyzer, FisherLDAAnalyzer, ResidualVectorLoader


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run Fisher LDA analysis for abstention direction research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis with default settings
  python scripts/run_fisher_analysis.py

  # Analyze specific layers only
  python scripts/run_fisher_analysis.py --layers 20 21 22 23 24

  # Use custom data directory and save results elsewhere
  python scripts/run_fisher_analysis.py --data-dir ./my_results --output-dir ./analysis_output

  # Load from complete file instead of layer files
  python scripts/run_fisher_analysis.py --complete-file ./results/residual_vectors_complete.pkl

  # Skip computation if results exist, just run evaluation
  python scripts/run_fisher_analysis.py --skip-computation --evaluate-only
        """,
    )

    # Data loading options
    data_group = parser.add_argument_group("Data Loading")
    data_group.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing residual vector data (default: auto-detect)",
    )
    data_group.add_argument(
        "--complete-file",
        type=Path,
        default=None,
        help="Load from complete file instead of layer files",
    )
    data_group.add_argument(
        "--run-timestamp",
        type=str,
        default=None,
        help="Specific run timestamp to load (latest if not specified)",
    )

    # Analysis options
    analysis_group = parser.add_argument_group("Analysis Options")
    analysis_group.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to analyze (all available if not specified)",
    )
    analysis_group.add_argument(
        "--lambda-shrinkage",
        type=float,
        default=-1.0,
        help="Shrinkage parameter for LDA (-1 for Ledoit-Wolf estimation)",
    )
    analysis_group.add_argument(
        "--alpha", type=float, default=1.0, help="Alpha parameter for shrinkage target"
    )
    analysis_group.add_argument(
        "--skip-computation",
        action="store_true",
        help="Skip LDA computation, load existing results",
    )
    analysis_group.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only run evaluation, skip LDA computation",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save results (default: ./results/LDA)",
    )
    output_group.add_argument(
        "--save-summary", action="store_true", help="Save summary results to CSV"
    )
    output_group.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top layers to show in summary (default: 10)",
    )

    # General options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress most output"
    )

    args = parser.parse_args()

    # Setup logging
    if args.quiet:
        logging.disable(logging.CRITICAL)
    else:
        setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    try:
        # Initialize data loader
        logger.info("Initializing data loader...")
        loader = ResidualVectorLoader(results_dir=args.data_dir)

        # Load residual vectors
        logger.info("Loading residual vectors...")
        if args.complete_file:
            residual_vectors = loader.load_complete_file(args.complete_file)
        else:
            residual_vectors = loader.load_layer_files(
                data_dir=args.data_dir, run_timestamp=args.run_timestamp
            )

        if not args.evaluate_only:
            # Initialize Fisher LDA analyzer
            logger.info("Initializing Fisher LDA analyzer...")
            lda_analyzer = FisherLDAAnalyzer(
                results_dir=args.output_dir or Path("./results/LDA"),
                lambda_=args.lambda_shrinkage,
                alpha=args.alpha,
            )

            # Load existing results if available
            if not args.skip_computation:
                lda_analyzer.load_existing_results()

            # Compute LDA directions
            if not args.skip_computation:
                logger.info("Computing Fisher LDA directions...")
                lda_directions = lda_analyzer.compute_directions(
                    residual_vectors=residual_vectors,
                    layers=args.layers,
                    save_incremental=True,
                )
            else:
                logger.info("Loading existing LDA directions...")
                lda_analyzer.load_existing_results()
                lda_directions = lda_analyzer.get_directions()

            # lda_metadata = lda_analyzer.get_metadata()  # Currently unused
        else:
            # Load existing LDA results for evaluation only
            logger.info("Loading existing LDA results for evaluation...")
            results_dir = args.output_dir or Path("./results/LDA")
            lda_files = list(results_dir.glob("lda_results_*.pkl"))

            if not lda_files:
                logger.error("No existing LDA results found for evaluation")
                return 1

            import pickle

            latest_file = max(lda_files, key=lambda f: f.stat().st_mtime)
            with open(latest_file, "rb") as f:
                lda_data = pickle.load(f)

            lda_directions = lda_data["lda_directions"]
            # lda_metadata = lda_data.get("lda_metadata", {})  # Currently unused

        # Initialize direction analyzer
        logger.info("Initializing direction analyzer...")
        direction_analyzer = DirectionAnalyzer()

        # Evaluate all directions
        logger.info("Evaluating directions...")
        evaluations = direction_analyzer.evaluate_all_layers(
            residual_vectors=residual_vectors,
            lda_directions=lda_directions,
            layers=args.layers,
        )

        if not evaluations:
            logger.error("No successful evaluations")
            return 1

        # Print summary
        if not args.quiet:
            direction_analyzer.print_summary(args.top_n)

        # Save summary if requested
        if args.save_summary:
            summary_df = direction_analyzer.get_summary_dataframe()
            if len(summary_df) > 0:
                output_dir = args.output_dir or Path("./results/LDA")
                output_dir.mkdir(parents=True, exist_ok=True)

                summary_file = output_dir / "analysis_summary.csv"
                summary_df.to_csv(summary_file, index=False)
                logger.info(f"Saved summary to {summary_file}")

        # Report best layer
        best_layer = direction_analyzer.get_best_layer()
        if best_layer is not None:
            best_eval = direction_analyzer.get_layer_evaluation(best_layer)
            if not args.quiet:
                print(f"\n{'='*50}")
                print(f"BEST LAYER: {best_layer}")
                print(f"AUC: {best_eval['auc']:.4f}")
                print(f"Cohen's d: {best_eval['cohen_d']:.4f}")
                print(f"Separation: {best_eval['separation']:.4f}")
                print(
                    f"Examples: {best_eval['n_pos']} positive, {best_eval['n_neg']} negative"
                )
                print(f"{'='*50}")

        logger.info("Analysis completed successfully")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
