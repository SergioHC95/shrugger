"""
Metrics analysis utilities for abstention experiments.

This module provides functionality for:
- Loading and processing metrics data from experiment runs
- Analyzing metrics by different grouping factors (form, label type)
- Visualizing metrics distributions and comparisons
- Finding and analyzing examples with extreme metric values
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import the save_figure function from plots module
from ..plots import save_figure

logger = logging.getLogger(__name__)


class MetricsAnalyzer:
    """
    A class for analyzing abstention metrics from experiment runs.

    This class handles:
    - Loading metrics data from CSV files
    - Computing statistics across different grouping factors
    - Visualizing metric distributions and comparisons
    - Finding examples with extreme metric values
    """

    def __init__(self, metrics_dir: Optional[Path] = None):
        """
        Initialize the metrics analyzer.

        Args:
            metrics_dir: Path to the metrics directory (default: ./results/metrics_analysis)
        """
        if metrics_dir is None:
            metrics_dir = Path("./results/metrics_analysis")

        self.metrics_dir = metrics_dir
        self.metrics_df = None
        self.form_df = None
        self.label_df = None
        self.latest_run = None

    def load_data(self, run_dir: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
        """
        Load metrics data from a specific run or the latest run.

        Args:
            run_dir: Specific run directory name (if None, uses the latest run)

        Returns:
            Tuple of (metrics_dataframe, run_directory_name)

        Raises:
            FileNotFoundError: If no metrics data is found
        """
        # Find all run directories
        run_dirs = [
            d
            for d in self.metrics_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ]

        if not run_dirs:
            raise FileNotFoundError(
                "No metrics data found. Please run the run_metrics_analysis.py script first."
            )

        # Use specified run or the latest one
        if run_dir is not None:
            run_path = self.metrics_dir / run_dir
            if not run_path.exists():
                available_runs = [d.name for d in run_dirs]
                raise FileNotFoundError(
                    f"Run {run_dir} not found. Available runs: {available_runs}"
                )
            self.latest_run = run_path
        else:
            self.latest_run = sorted(run_dirs)[-1]

        # Load the metrics data
        metrics_file = self.latest_run / "all_metrics.csv"
        self.metrics_df = pd.read_csv(metrics_file)

        logger.info(
            f"Loaded {len(self.metrics_df)} metric records from {self.latest_run.name}"
        )

        # Load summary files if they exist
        self._load_summary_files()

        return self.metrics_df, self.latest_run.name

    def _load_summary_files(self):
        """Load form and label type summary files if they exist."""
        try:
            form_summary_file = self.latest_run / "form_summary.csv"
            if form_summary_file.exists():
                self.form_df = pd.read_csv(form_summary_file)
                logger.info(f"Loaded form summary with {len(self.form_df)} entries")
        except Exception as e:
            logger.warning(f"Could not load form summary: {e}")

        try:
            label_summary_file = self.latest_run / "label_type_summary.csv"
            if label_summary_file.exists():
                self.label_df = pd.read_csv(label_summary_file)
                logger.info(
                    f"Loaded label type summary with {len(self.label_df)} entries"
                )
        except Exception as e:
            logger.warning(f"Could not load label type summary: {e}")

    def compute_metrics_by_form(self) -> pd.DataFrame:
        """
        Compute metrics statistics grouped by form.

        Returns:
            DataFrame with metrics statistics by form
        """
        if self.metrics_df is None:
            raise ValueError("No metrics data loaded. Call load_data() first.")

        # Group metrics by form and permutation to get per-experiment metrics
        if "permutation" in self.metrics_df.columns:
            exp_form_metrics = self.metrics_df.groupby(["form", "permutation"])
        else:
            # If permutation is not available, try to extract it from experiment_name
            if "experiment_name" in self.metrics_df.columns:
                # Extract permutation from experiment_name (assuming format like "V0_alpha_p1")
                self.metrics_df["permutation"] = self.metrics_df[
                    "experiment_name"
                ].str.extract(r"p(\d+)")
                exp_form_metrics = self.metrics_df.groupby(["form", "permutation"])
            else:
                # If neither is available, just use form
                logger.warning(
                    "No permutation or experiment_name column found. Using only form for grouping."
                )
                exp_form_metrics = self.metrics_df.groupby(["form"])

        # Calculate statistics for CA score
        ca_stats = (
            exp_form_metrics["ca_score"].agg(["mean", "std", "count"]).reset_index()
        )
        ca_stats["se"] = ca_stats["std"] / np.sqrt(ca_stats["count"])
        ca_stats = ca_stats.rename(columns={"mean": "ca_score_mean"})

        # Calculate statistics for HEDGE score
        hedge_stats = (
            exp_form_metrics["hedge_score"].agg(["mean", "std", "count"]).reset_index()
        )
        hedge_stats["se"] = hedge_stats["std"] / np.sqrt(hedge_stats["count"])
        hedge_stats = hedge_stats.rename(columns={"mean": "hedge_score_mean"})

        # Merge the statistics
        if "permutation" in ca_stats.columns:
            form_stats = (
                ca_stats.groupby("form")
                .agg({"ca_score_mean": "mean", "se": "mean", "count": "sum"})
                .reset_index()
            )

            form_hedge_stats = (
                hedge_stats.groupby("form")
                .agg({"hedge_score_mean": "mean", "se": "mean"})
                .reset_index()
            )

            form_stats = pd.merge(form_stats, form_hedge_stats, on="form")
            form_stats = form_stats.rename(
                columns={"se_x": "ca_score_se", "se_y": "hedge_score_se"}
            )
        else:
            form_stats = pd.merge(
                ca_stats[["form", "ca_score_mean", "se", "count"]],
                hedge_stats[["form", "hedge_score_mean", "se"]],
                on="form",
                suffixes=("_ca", "_hedge"),
            )

        # Sort by form
        form_stats = form_stats.sort_values("form")

        return form_stats

    def compute_metrics_by_label_type(self) -> pd.DataFrame:
        """
        Compute metrics statistics grouped by label type.

        Returns:
            DataFrame with metrics statistics by label type
        """
        if self.metrics_df is None:
            raise ValueError("No metrics data loaded. Call load_data() first.")

        # Group metrics by label_type and permutation to get per-experiment metrics
        if "permutation" in self.metrics_df.columns:
            exp_label_metrics = self.metrics_df.groupby(["label_type", "permutation"])
        else:
            # If permutation is not available, try to extract it from experiment_name
            if "experiment_name" in self.metrics_df.columns:
                # Extract permutation from experiment_name (assuming format like "V0_alpha_p1")
                if "permutation" not in self.metrics_df.columns:
                    self.metrics_df["permutation"] = self.metrics_df[
                        "experiment_name"
                    ].str.extract(r"p(\d+)")
                exp_label_metrics = self.metrics_df.groupby(
                    ["label_type", "permutation"]
                )
            else:
                # If neither is available, just use label_type
                logger.warning(
                    "No permutation or experiment_name column found. Using only label_type for grouping."
                )
                exp_label_metrics = self.metrics_df.groupby(["label_type"])

        # Calculate statistics for CA score
        ca_stats = (
            exp_label_metrics["ca_score"].agg(["mean", "std", "count"]).reset_index()
        )
        ca_stats["se"] = ca_stats["std"] / np.sqrt(ca_stats["count"])
        ca_stats = ca_stats.rename(columns={"mean": "ca_score_mean"})

        # Calculate statistics for HEDGE score
        hedge_stats = (
            exp_label_metrics["hedge_score"].agg(["mean", "std", "count"]).reset_index()
        )
        hedge_stats["se"] = hedge_stats["std"] / np.sqrt(hedge_stats["count"])
        hedge_stats = hedge_stats.rename(columns={"mean": "hedge_score_mean"})

        # Merge the statistics
        if "permutation" in ca_stats.columns:
            label_stats = (
                ca_stats.groupby("label_type")
                .agg({"ca_score_mean": "mean", "se": "mean", "count": "sum"})
                .reset_index()
            )

            label_hedge_stats = (
                hedge_stats.groupby("label_type")
                .agg({"hedge_score_mean": "mean", "se": "mean"})
                .reset_index()
            )

            label_stats = pd.merge(label_stats, label_hedge_stats, on="label_type")
            label_stats = label_stats.rename(
                columns={"se_x": "ca_score_se", "se_y": "hedge_score_se"}
            )
        else:
            label_stats = pd.merge(
                ca_stats[["label_type", "ca_score_mean", "se", "count"]],
                hedge_stats[["label_type", "hedge_score_mean", "se"]],
                on="label_type",
                suffixes=("_ca", "_hedge"),
            )

        return label_stats

    def plot_ca_scores_by_form(
        self,
        form_stats: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (12, 6),
        save: bool = True,
        filename: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot CA scores by form with error bars.

        Args:
            form_stats: DataFrame with form statistics (if None, computes it)
            figsize: Figure size as (width, height)
            save: Whether to save the figure
            filename: Filename to use when saving (default: auto-generated)

        Returns:
            The matplotlib Figure object or the path to the saved figure if save=True
        """
        if form_stats is None:
            form_stats = self.compute_metrics_by_form()

        fig, ax = plt.subplots(figsize=figsize)

        # Plot CA scores without error bars (we'll add them manually)
        bars = sns.barplot(
            x="form",
            y="ca_score_mean",
            data=form_stats,
            color="skyblue",
            errorbar=None,
            ax=ax,
        )

        # Add custom error bars
        for i, row in form_stats.iterrows():
            x = i
            y = row["ca_score_mean"]
            se = row.get("ca_score_se", row.get("se_ca", row.get("se")))

            if not pd.isna(se) and se > 0:
                ax.errorbar(
                    x, y, yerr=se, fmt="none", color="black", capsize=3, elinewidth=1
                )

        # Add labels and title
        ax.set_title("Average Confident-Abstain (CA) Score by Prompt Form", fontsize=16)
        ax.set_xlabel("Prompt Form", fontsize=14)
        ax.set_ylabel("Average CA Score", fontsize=14)

        # Calculate appropriate y-axis limits based on data
        max_value = form_stats["ca_score_mean"].max()
        y_max = max_value * 1.3  # Add 30% padding above the max value
        ax.set_ylim(0, y_max)

        # Add value labels on top of bars
        for i, v in enumerate(form_stats["ca_score_mean"]):
            ax.text(i, v + (y_max * 0.02), f"{v:.3f}", ha="center", fontsize=12)

        plt.tight_layout()
        
        # Save figure if requested
        if save:
            if filename is None:
                filename = f"ca_scores_by_form"
            return save_figure(fig=fig, filename=filename)
            
        return fig

    def plot_hedge_scores_by_form(
        self,
        form_stats: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (12, 6),
        save: bool = True,
        filename: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot HEDGE scores by form with error bars.

        Args:
            form_stats: DataFrame with form statistics (if None, computes it)
            figsize: Figure size as (width, height)
            save: Whether to save the figure
            filename: Filename to use when saving (default: auto-generated)

        Returns:
            The matplotlib Figure object or the path to the saved figure if save=True
        """
        if form_stats is None:
            form_stats = self.compute_metrics_by_form()

        fig, ax = plt.subplots(figsize=figsize)

        # Plot HEDGE scores without error bars (we'll add them manually)
        bars = sns.barplot(
            x="form",
            y="hedge_score_mean",
            data=form_stats,
            color="lightgreen",
            errorbar=None,
            ax=ax,
        )

        # Add custom error bars
        for i, row in form_stats.iterrows():
            x = i
            y = row["hedge_score_mean"]
            se = row.get("hedge_score_se", row.get("se_hedge", row.get("se")))

            if not pd.isna(se) and se > 0:
                ax.errorbar(
                    x, y, yerr=se, fmt="none", color="black", capsize=3, elinewidth=1
                )

        # Add labels and title
        ax.set_title("Average Hedging (HEDGE) Score by Prompt Form", fontsize=16)
        ax.set_xlabel("Prompt Form", fontsize=14)
        ax.set_ylabel("Average HEDGE Score", fontsize=14)

        # Calculate appropriate y-axis limits based on data
        max_value = form_stats["hedge_score_mean"].max()
        y_max = max_value * 1.3  # Add 30% padding above the max value
        ax.set_ylim(0, y_max)

        # Add value labels on top of bars
        for i, v in enumerate(form_stats["hedge_score_mean"]):
            ax.text(i, v + (y_max * 0.02), f"{v:.3f}", ha="center", fontsize=12)

        plt.tight_layout()
        
        # Save figure if requested
        if save:
            if filename is None:
                filename = f"hedge_scores_by_form"
            return save_figure(fig=fig, filename=filename)
            
        return fig

    def plot_metrics_by_label_type(
        self,
        label_stats: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (10, 6),
        save: bool = True,
        filename: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot CA and HEDGE scores by label type with error bars.

        Args:
            label_stats: DataFrame with label type statistics (if None, computes it)
            figsize: Figure size as (width, height)
            save: Whether to save the figure
            filename: Filename to use when saving (default: auto-generated)

        Returns:
            The matplotlib Figure object or the path to the saved figure if save=True
        """
        if label_stats is None:
            label_stats = self.compute_metrics_by_label_type()

        fig, ax = plt.subplots(figsize=figsize)

        # Set up positions for grouped bars
        x = np.arange(len(label_stats["label_type"]))
        width = 0.35

        # Create bars
        ca_bars = ax.bar(
            x - width / 2,
            label_stats["ca_score_mean"],
            width,
            label="CA Score",
            color="skyblue",
        )
        hedge_bars = ax.bar(
            x + width / 2,
            label_stats["hedge_score_mean"],
            width,
            label="HEDGE Score",
            color="lightgreen",
        )

        # Add error bars
        for i, row in label_stats.iterrows():
            # CA score error bar
            x_pos = i - width / 2
            y = row["ca_score_mean"]
            se = row.get("ca_score_se", row.get("se_ca"))
            if not pd.isna(se) and se > 0:
                ax.errorbar(
                    x_pos,
                    y,
                    yerr=se,
                    fmt="none",
                    color="black",
                    capsize=3,
                    elinewidth=1,
                )

            # HEDGE score error bar
            x_pos = i + width / 2
            y = row["hedge_score_mean"]
            se = row.get("hedge_score_se", row.get("se_hedge"))
            if not pd.isna(se) and se > 0:
                ax.errorbar(
                    x_pos,
                    y,
                    yerr=se,
                    fmt="none",
                    color="black",
                    capsize=3,
                    elinewidth=1,
                )

        # Add labels and title
        ax.set_title("Average Metrics by Label Type", fontsize=16)
        ax.set_xlabel("Label Type", fontsize=14)
        ax.set_ylabel("Average Score", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(label_stats["label_type"])

        # Calculate appropriate y-axis limits based on data
        max_ca = label_stats["ca_score_mean"].max()
        max_hedge = label_stats["hedge_score_mean"].max()
        max_value = max(max_ca, max_hedge)
        y_max = max_value * 1.3  # Add 30% padding above the max value
        ax.set_ylim(0, y_max)

        ax.legend()

        # Add value labels on top of bars
        for bar in ca_bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (y_max * 0.02),
                f"{height:.3f}",
                ha="center",
                fontsize=12,
            )

        for bar in hedge_bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (y_max * 0.02),
                f"{height:.3f}",
                ha="center",
                fontsize=12,
            )

        plt.tight_layout()
        
        # Save figure if requested
        if save:
            if filename is None:
                filename = f"metrics_by_label_type"
            return save_figure(fig=fig, filename=filename)
            
        return fig

    def plot_metric_distributions(
        self, figsize: Tuple[int, int] = (16, 6), save: bool = True, filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot histograms of CA and HEDGE score distributions.

        Args:
            figsize: Figure size as (width, height)
            save: Whether to save the figure
            filename: Filename to use when saving (default: auto-generated)

        Returns:
            The matplotlib Figure object or the path to the saved figure if save=True
        """
        if self.metrics_df is None:
            raise ValueError("No metrics data loaded. Call load_data() first.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # CA score histogram
        sns.histplot(
            self.metrics_df["ca_score"], bins=30, kde=True, ax=ax1, color="skyblue"
        )
        ax1.set_title("Distribution of Confident-Abstain (CA) Scores", fontsize=16)
        ax1.set_xlabel("CA Score", fontsize=14)
        ax1.set_ylabel("Frequency", fontsize=14)

        # Set x-axis limit for CA scores based on data
        ca_max = self.metrics_df["ca_score"].max()
        ca_99th_percentile = self.metrics_df["ca_score"].quantile(
            0.99
        )  # Use 99th percentile to avoid outliers
        ax1.set_xlim(0, min(ca_max * 1.1, ca_99th_percentile * 1.3))

        # HEDGE score histogram
        sns.histplot(
            self.metrics_df["hedge_score"],
            bins=30,
            kde=True,
            ax=ax2,
            color="lightgreen",
        )
        ax2.set_title("Distribution of Hedging (HEDGE) Scores", fontsize=16)
        ax2.set_xlabel("HEDGE Score", fontsize=14)
        ax2.set_ylabel("Frequency", fontsize=14)

        # Set x-axis limit for HEDGE scores based on data
        hedge_max = self.metrics_df["hedge_score"].max()
        hedge_99th_percentile = self.metrics_df["hedge_score"].quantile(
            0.99
        )  # Use 99th percentile to avoid outliers
        ax2.set_xlim(0, min(hedge_max * 1.1, hedge_99th_percentile * 1.3))

        plt.tight_layout()
        
        # Save figure if requested
        if save:
            if filename is None:
                filename = f"metric_distributions"
            return save_figure(fig=fig, filename=filename)
            
        return fig

    def find_top_examples_by_metric(
        self, metric: str = "hedge_score", n: int = 5
    ) -> pd.DataFrame:
        """
        Find top examples with highest values for a given metric.

        Args:
            metric: Metric to sort by ('hedge_score' or 'ca_score')
            n: Number of examples to return

        Returns:
            DataFrame with top examples
        """
        if self.metrics_df is None:
            raise ValueError("No metrics data loaded. Call load_data() first.")

        if metric not in self.metrics_df.columns:
            raise ValueError(f"Metric '{metric}' not found in data")

        # Sort by the specified metric in descending order
        top_examples = self.metrics_df.sort_values(metric, ascending=False).head(n)
        return top_examples

    def analyze_hedge_components(self, example_row: pd.Series) -> Dict[str, float]:
        """
        Analyze the components of the HEDGE score for a specific example.

        Args:
            example_row: A row from the metrics DataFrame

        Returns:
            Dictionary with HEDGE score components
        """
        try:
            # Convert the JSON string to a dictionary
            probs = json.loads(example_row["canonical_probs"])

            # Calculate Yes/No probabilities for clarity
            p_yes = probs.get("YY", 0) + probs.get("Y", 0)
            p_no = probs.get("N", 0) + probs.get("NN", 0)
            p_idk = probs.get("A", 0)

            # Calculate the HEDGE score components
            s = 1.0 - p_idk

            result = {
                "p_yes": p_yes,
                "p_no": p_no,
                "p_idk": p_idk,
                "s": s,
            }

            if s > 0:
                r_yes = p_yes / s
                r_no = p_no / s
                result["r_yes"] = r_yes
                result["r_no"] = r_no

                # Calculate entropy
                entropy = 0
                if r_yes > 0:
                    entropy -= r_yes * np.log(r_yes)
                if r_no > 0:
                    entropy -= r_no * np.log(r_no)
                normalized_entropy = entropy / np.log(2)
                result["normalized_entropy"] = normalized_entropy
                result["hedge_score_calc"] = s * normalized_entropy

            return result

        except Exception as e:
            logger.error(f"Could not analyze HEDGE components: {e}")
            return {}


def load_metrics_data(
    metrics_dir: Optional[Path] = None, run_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Load metrics data from a specific run or the latest run.

    Args:
        metrics_dir: Path to the metrics directory (default: ./results/metrics_analysis)
        run_dir: Specific run directory name (if None, uses the latest run)

    Returns:
        Tuple of (metrics_dataframe, run_directory_name)
    """
    analyzer = MetricsAnalyzer(metrics_dir)
    return analyzer.load_data(run_dir)


def plot_metrics_by_form(
    metrics_df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6), save: bool = True, filename: Optional[str] = None
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Create bar plots of CA and HEDGE scores by form.

    Args:
        metrics_df: DataFrame with metrics data
        figsize: Figure size as (width, height)
        save: Whether to save the figures
        filename: Base filename to use when saving (default: auto-generated)

    Returns:
        Tuple of (ca_figure, hedge_figure) or paths to saved figures if save=True
    """
    analyzer = MetricsAnalyzer()
    analyzer.metrics_df = metrics_df

    form_stats = analyzer.compute_metrics_by_form()
    
    ca_filename = None
    hedge_filename = None
    
    if filename is not None:
        ca_filename = f"{filename}_ca_scores"
        hedge_filename = f"{filename}_hedge_scores"
        
    ca_fig = analyzer.plot_ca_scores_by_form(form_stats, figsize, save, ca_filename)
    hedge_fig = analyzer.plot_hedge_scores_by_form(form_stats, figsize, save, hedge_filename)

    return ca_fig, hedge_fig


def plot_metrics_by_label_type(
    metrics_df: pd.DataFrame, figsize: Tuple[int, int] = (10, 6), save: bool = True, filename: Optional[str] = None
) -> plt.Figure:
    """
    Create a grouped bar plot for CA and HEDGE scores by label type.

    Args:
        metrics_df: DataFrame with metrics data
        figsize: Figure size as (width, height)
        save: Whether to save the figure
        filename: Filename to use when saving (default: auto-generated)

    Returns:
        The matplotlib Figure object or the path to the saved figure if save=True
    """
    analyzer = MetricsAnalyzer()
    analyzer.metrics_df = metrics_df

    label_stats = analyzer.compute_metrics_by_label_type()
    return analyzer.plot_metrics_by_label_type(label_stats, figsize, save, filename)


def plot_metric_distributions(
    metrics_df: pd.DataFrame, figsize: Tuple[int, int] = (16, 6), save: bool = True, filename: Optional[str] = None
) -> plt.Figure:
    """
    Plot histograms of CA and HEDGE score distributions.

    Args:
        metrics_df: DataFrame with metrics data
        figsize: Figure size as (width, height)
        save: Whether to save the figure
        filename: Filename to use when saving (default: auto-generated)

    Returns:
        The matplotlib Figure object or the path to the saved figure if save=True
    """
    analyzer = MetricsAnalyzer()
    analyzer.metrics_df = metrics_df
    return analyzer.plot_metric_distributions(figsize, save, filename)


def find_top_examples(
    metrics_df: pd.DataFrame, metric: str = "hedge_score", n: int = 5
) -> pd.DataFrame:
    """
    Find top examples with highest values for a given metric.

    Args:
        metrics_df: DataFrame with metrics data
        metric: Metric to sort by ('hedge_score' or 'ca_score')
        n: Number of examples to return

    Returns:
        DataFrame with top examples
    """
    analyzer = MetricsAnalyzer()
    analyzer.metrics_df = metrics_df
    return analyzer.find_top_examples_by_metric(metric, n)
