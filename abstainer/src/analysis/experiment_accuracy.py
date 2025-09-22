"""
Experiment accuracy analysis for abstention research.

This module provides functionality for:
- Loading and processing experiment results
- Analyzing accuracy by difficulty and question type
- Visualizing accuracy patterns
- Generating summary statistics
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from ..plots import save_figure

logger = logging.getLogger(__name__)


def find_experiment_dirs(results_dir: str = "./results") -> List[Path]:
    """
    Find all experiment directories in the comprehensive_experiments directory.

    Args:
        results_dir: Base directory for results

    Returns:
        List of experiment directory paths
    """
    experiment_dirs = []
    comp_exp_dir = Path(results_dir) / "comprehensive_experiments"

    # Check if comprehensive experiments directory exists
    if comp_exp_dir.exists() and comp_exp_dir.is_dir():
        # Look for run_* directories
        for run_dir in comp_exp_dir.glob("run_*"):
            if run_dir.is_dir():
                # Add all experiment directories within this run
                for exp_dir in run_dir.glob("*"):
                    if exp_dir.is_dir() and any(exp_dir.glob("likert_results_*.json")):
                        experiment_dirs.append(exp_dir)

    return experiment_dirs


def load_experiment_results(exp_dir: Path) -> Optional[Tuple[Dict, Dict]]:
    """
    Load results from an experiment directory.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        Tuple of (results, experiment_info) or None if loading fails
    """
    results_files = list(exp_dir.glob("likert_results_*.json"))
    if not results_files:
        return None

    # Load the first results file found
    results_file = results_files[0]
    form = results_file.stem.replace("likert_results_", "")

    try:
        with open(results_file) as f:
            results = json.load(f)

        # Load experiment metadata from summary.json if available
        summary_file = exp_dir / "summary.json"
        experiment_info = {}
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
                experiment_info = {
                    'experiment_name': summary.get('experiment_name', ''),
                    'form': summary.get('form', form),
                    'label_type': summary.get('label_type', ''),
                    'permutation': summary.get('permutation', '')
                }
        else:
            # Extract information from directory name if no summary file
            experiment_info = {
                'experiment_name': exp_dir.name,
                'form': form,
                'label_type': '',
                'permutation': ''
            }

        return results, experiment_info
    except Exception as e:
        logger.error(f"Error loading results from {exp_dir}: {str(e)}")
        return None


def process_experiment_results(experiment_dirs: List[Path], 
                              show_progress: bool = True) -> pd.DataFrame:
    """
    Process all experiments and collect results into a DataFrame.

    Args:
        experiment_dirs: List of experiment directory paths
        show_progress: Whether to show a progress bar

    Returns:
        DataFrame with processed results
    """
    all_results = []
    
    iterator = tqdm(experiment_dirs, desc="Loading experiments") if show_progress else experiment_dirs
    
    for exp_dir in iterator:
        result = load_experiment_results(exp_dir)
        if result:
            results, experiment_info = result

            # Process each question result
            for question_id, question_result in results.items():
                # Extract basic information
                question_text = question_result.get('question', '')
                answer = question_result.get('answer', '')
                subject = question_result.get('subject', '')
                difficulty = question_result.get('difficulty', '')

                # Extract prediction information
                pred_label = question_result.get('pred_label', '')
                canonical_label = question_result.get('canonical_label', '')
                score = question_result.get('score', 0)
                is_valid = question_result.get('is_valid', False)

                # Determine if prediction is correct
                # Score of 2 means definitely correct, 1 means probably correct
                is_correct = score >= 1

                # Determine question type (Yes/No/Unanswerable)
                question_type = 'Unanswerable'
                if answer.lower() == 'yes':
                    question_type = 'Yes'
                elif answer.lower() == 'no':
                    question_type = 'No'

                # Add to results list
                all_results.append({
                    'experiment_name': experiment_info['experiment_name'],
                    'form': experiment_info['form'],
                    'label_type': experiment_info['label_type'],
                    'permutation': experiment_info['permutation'],
                    'question_id': question_id,
                    'question': question_text,
                    'answer': answer,
                    'question_type': question_type,
                    'subject': subject,
                    'difficulty': difficulty,
                    'pred_label': pred_label,
                    'canonical_label': canonical_label,
                    'score': score,
                    'is_valid': is_valid,
                    'is_correct': is_correct
                })

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Convert difficulty to numeric if it's not already
    if results_df['difficulty'].dtype == 'object':
        try:
            results_df['difficulty_num'] = pd.to_numeric(results_df['difficulty'])
        except:
            # If conversion fails, create a categorical mapping
            difficulty_map = {d: i+1 for i, d in enumerate(sorted(results_df['difficulty'].unique()))}
            results_df['difficulty_num'] = results_df['difficulty'].map(difficulty_map)
    
    return results_df


def calculate_accuracy_statistics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate accuracy statistics by difficulty and question type.

    Args:
        results_df: DataFrame with processed results

    Returns:
        Dictionary with accuracy statistics
    """
    # Calculate accuracy by difficulty
    accuracy_by_difficulty = results_df.groupby('difficulty')['is_correct'].agg(['mean', 'count']).reset_index()
    
    # Calculate accuracy by difficulty and question type
    accuracy_by_difficulty_type = results_df.groupby(['difficulty', 'question_type'])['is_correct'].agg(['mean', 'count']).reset_index()
    
    # Calculate accuracy by question type
    question_type_accuracy = results_df.groupby('question_type')['is_correct'].agg(['mean', 'count']).reset_index()
    
    # Calculate overall accuracy
    overall_accuracy = results_df['is_correct'].mean()
    
    # Find best and worst performing categories
    best_difficulty = accuracy_by_difficulty.loc[accuracy_by_difficulty['mean'].idxmax()]
    worst_difficulty = accuracy_by_difficulty.loc[accuracy_by_difficulty['mean'].idxmin()]
    best_question_type = question_type_accuracy.loc[question_type_accuracy['mean'].idxmax()]
    worst_question_type = question_type_accuracy.loc[question_type_accuracy['mean'].idxmin()]
    
    return {
        'overall_accuracy': overall_accuracy,
        'accuracy_by_difficulty': accuracy_by_difficulty,
        'accuracy_by_difficulty_type': accuracy_by_difficulty_type,
        'question_type_accuracy': question_type_accuracy,
        'best_difficulty': best_difficulty,
        'worst_difficulty': worst_difficulty,
        'best_question_type': best_question_type,
        'worst_question_type': worst_question_type,
        'total_questions': len(results_df),
        'num_experiments': results_df['experiment_name'].nunique()
    }


def plot_accuracy_by_difficulty(results_df: pd.DataFrame, 
                               figsize: Tuple[int, int] = (10, 6),
                               save: bool = True,
                               filename: Optional[str] = None) -> Tuple:
    """
    Plot accuracy by difficulty level.

    Args:
        results_df: DataFrame with processed results
        figsize: Figure size as (width, height)
        save: Whether to save the figure
        filename: Filename to use when saving (default: auto-generated)

    Returns:
        Matplotlib figure and axes
    """
    plt.figure(figsize=figsize)

    # First, calculate accuracy for each experiment (form+permutation) by difficulty
    exp_difficulty_stats = results_df.groupby(['experiment_name', 'difficulty'])['is_correct'].mean().reset_index()

    # Then aggregate across experiments to get mean and std for each difficulty
    difficulty_stats = exp_difficulty_stats.groupby('difficulty')['is_correct'].agg(['mean', 'std', 'count']).reset_index()

    # Calculate standard error
    difficulty_stats['se'] = difficulty_stats['std'] / np.sqrt(difficulty_stats['count'])

    # Create the bar plot without error bars
    ax = sns.barplot(x='difficulty', y='mean', hue='difficulty', data=difficulty_stats,
                    palette='viridis', legend=False, errorbar=None)

    # Add custom error bars
    for i, row in difficulty_stats.iterrows():
        bar = ax.patches[i]
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()

        se = float(row['se'])

        if se > 0:
            ax.errorbar(
                x, y,
                yerr=se,
                fmt='none', color='black', capsize=3, elinewidth=1
            )

    plt.title('Model Accuracy by Difficulty Level', fontsize=16)
    plt.xlabel('Difficulty Level', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    fig = plt.gcf()
    
    # Save figure if requested
    if save:
        if filename is None:
            filename = "accuracy_by_difficulty"
        save_figure(fig=fig, filename=filename)
    
    return fig, ax


def plot_accuracy_by_question_type_and_difficulty(results_df: pd.DataFrame,
                                                figsize: Tuple[int, int] = (12, 7),
                                                save: bool = True,
                                                filename: Optional[str] = None) -> Tuple:
    """
    Plot accuracy by question type and difficulty.

    Args:
        results_df: DataFrame with processed results
        figsize: Figure size as (width, height)
        save: Whether to save the figure
        filename: Filename to use when saving (default: auto-generated)

    Returns:
        Matplotlib figure and axes
    """
    plt.figure(figsize=figsize)

    # Define question type order for consistent colors
    question_type_order = ['Yes', 'No', 'Unanswerable']

    # First, calculate accuracy for each experiment (form+permutation) by difficulty and question type
    exp_difficulty_type_stats = results_df.groupby(['experiment_name', 'difficulty', 'question_type'])['is_correct'].mean().reset_index()

    # Then aggregate across experiments to get mean and std for each difficulty and question type
    difficulty_type_stats = exp_difficulty_type_stats.groupby(['difficulty', 'question_type'])['is_correct'].agg(['mean', 'std', 'count']).reset_index()

    # Calculate standard error
    difficulty_type_stats['se'] = difficulty_type_stats['std'] / np.sqrt(difficulty_type_stats['count'])

    # Create the bar plot without error bars
    ax = sns.barplot(x='difficulty', y='mean', hue='question_type', data=difficulty_type_stats,
                    palette='Set2', hue_order=question_type_order, errorbar=None)

    # Get the number of hue categories
    n_hue = len(question_type_order)

    # Add custom error bars
    for i, row in difficulty_type_stats.iterrows():
        # Calculate the patch index based on difficulty and question type
        diff_idx = list(difficulty_type_stats['difficulty'].unique()).index(row['difficulty'])
        qt_idx = question_type_order.index(row['question_type'])
        patch_idx = diff_idx * n_hue + qt_idx

        # Make sure we don't exceed the number of patches
        if patch_idx < len(ax.patches):
            bar = ax.patches[patch_idx]
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()

            se = float(row['se'])

            if se > 0:
                ax.errorbar(
                    x, y,
                    yerr=se,
                    fmt='none', color='black', capsize=3, elinewidth=1
                )

    plt.title('Model Accuracy by Question Type and Difficulty', fontsize=16)
    plt.xlabel('Difficulty Level', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Question Type', title_fontsize=12, fontsize=10)
    plt.tight_layout()
    
    fig = plt.gcf()
    
    # Save figure if requested
    if save:
        if filename is None:
            filename = "accuracy_by_question_type_and_difficulty"
        save_figure(fig=fig, filename=filename)
    
    return fig, ax


def generate_performance_summary(stats: Dict[str, Any]) -> str:
    """
    Generate a text summary of model performance.

    Args:
        stats: Dictionary with accuracy statistics from calculate_accuracy_statistics

    Returns:
        Formatted text summary
    """
    overall_accuracy = stats['overall_accuracy']
    question_type_accuracy = stats['question_type_accuracy']
    best_difficulty = stats['best_difficulty']
    worst_difficulty = stats['worst_difficulty']
    best_question_type = stats['best_question_type']
    worst_question_type = stats['worst_question_type']
    total_questions = stats['total_questions']
    num_experiments = stats['num_experiments']
    
    # Generate text summary
    summary = f"""
# Model Performance Summary

## Overall Performance
- Overall accuracy across all experiments: {overall_accuracy:.2%}
- Total questions analyzed: {total_questions}
- Number of experiments: {num_experiments}

## Performance by Question Type
- Yes questions: {question_type_accuracy[question_type_accuracy['question_type'] == 'Yes']['mean'].values[0]:.2%} accuracy ({question_type_accuracy[question_type_accuracy['question_type'] == 'Yes']['count'].values[0]} questions)
- No questions: {question_type_accuracy[question_type_accuracy['question_type'] == 'No']['mean'].values[0]:.2%} accuracy ({question_type_accuracy[question_type_accuracy['question_type'] == 'No']['count'].values[0]} questions)
- Unanswerable questions: {question_type_accuracy[question_type_accuracy['question_type'] == 'Unanswerable']['mean'].values[0]:.2%} accuracy ({question_type_accuracy[question_type_accuracy['question_type'] == 'Unanswerable']['count'].values[0]} questions)

## Performance by Difficulty
- Best performance on difficulty level {best_difficulty['difficulty']}: {best_difficulty['mean']:.2%} accuracy ({best_difficulty['count']} questions)
- Worst performance on difficulty level {worst_difficulty['difficulty']}: {worst_difficulty['mean']:.2%} accuracy ({worst_difficulty['count']} questions)

## Key Findings
- The model performs best on {best_question_type['question_type']} questions ({best_question_type['mean']:.2%} accuracy)
- The model struggles most with {worst_question_type['question_type']} questions ({worst_question_type['mean']:.2%} accuracy)
"""
    
    return summary


def plot_accuracy_by_form(results_df: pd.DataFrame,
                         figsize: Tuple[int, int] = (10, 6),
                         save: bool = True,
                         filename: Optional[str] = None) -> Tuple:
    """
    Plot accuracy by prompt form.

    Args:
        results_df: DataFrame with processed results
        figsize: Figure size as (width, height)
        save: Whether to save the figure
        filename: Filename to use when saving (default: auto-generated)

    Returns:
        Matplotlib figure and axes
    """
    plt.figure(figsize=figsize)
    
    # First, calculate accuracy for each experiment (form+permutation)
    exp_form_accuracy = results_df.groupby(['experiment_name', 'form', 'permutation'])['is_correct'].mean().reset_index()

    # Then aggregate across permutations to get mean and std for each form
    form_accuracy = exp_form_accuracy.groupby('form')['is_correct'].agg(['mean', 'std', 'count']).reset_index()

    # Calculate standard error
    form_accuracy['se'] = form_accuracy['std'] / np.sqrt(form_accuracy['count'])

    # Sort forms sequentially
    form_order = sorted(form_accuracy['form'].unique())
    form_accuracy = form_accuracy.sort_values('form', key=lambda x: [form_order.index(i) for i in x])

    # Create the bar plot without error bars
    ax = sns.barplot(x='form', y='mean', hue='form', data=form_accuracy,
                    palette='Blues_d', errorbar=None, legend=False, order=form_order)

    # Add custom error bars
    for i, row in form_accuracy.iterrows():
        # Get the index in the plot order
        plot_idx = form_order.index(row['form'])
        bar = ax.patches[plot_idx]
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()

        se = float(row['se'])

        if se > 0:
            ax.errorbar(
                x, y,
                yerr=se,
                fmt='none', color='black', capsize=3, elinewidth=1
            )

    plt.title('Model Accuracy by Prompt Form', fontsize=16)
    plt.xlabel('Form', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    fig = plt.gcf()
    
    # Save figure if requested
    if save:
        if filename is None:
            filename = "accuracy_by_form"
        save_figure(fig=fig, filename=filename)
    
    return fig, ax


def plot_accuracy_by_form_and_question_type(results_df: pd.DataFrame,
                                          figsize: Tuple[int, int] = (14, 8),
                                          save: bool = True,
                                          filename: Optional[str] = None) -> Tuple:
    """
    Plot accuracy by form and question type with permutation variation.

    Args:
        results_df: DataFrame with processed results
        figsize: Figure size as (width, height)
        save: Whether to save the figure
        filename: Filename to use when saving (default: auto-generated)

    Returns:
        Matplotlib figure and axes
    """
    # Define question type order
    question_type_order = ['Yes', 'No', 'Unanswerable']
    
    # Ensure base_form column exists
    if 'base_form' not in results_df.columns:
        results_df['base_form'] = results_df['form'].astype(str).str.extract(r'(V\d+)', expand=False)

    # Ensure permutation column exists
    if 'permutation' not in results_df.columns:
        if 'experiment_name' in results_df.columns:
            results_df['permutation'] = results_df['experiment_name'].astype(str).str.extract(r'p(\d+)', expand=False)
        else:
            # If no explicit permutation info, treat all rows as a single permutation group
            results_df['permutation'] = 'p0'

    # Basic sanitation
    # Keep only rows with necessary fields
    results_df = results_df.dropna(subset=['base_form', 'permutation', 'question_type', 'is_correct'])
    # Coerce correctness to numeric (0/1)
    results_df['is_correct'] = results_df['is_correct'].astype(float)

    # Calculate accuracy per (base_form, permutation, question_type)
    permutation_stats = (
        results_df
        .groupby(['base_form', 'permutation', 'question_type'], as_index=False)['is_correct']
        .agg(mean='mean', count='count')
    )

    # Aggregate across permutations for each base_form × question_type
    form_type_stats = (
        permutation_stats
        .groupby(['base_form', 'question_type'], as_index=False)['mean']
        .agg(avg_accuracy='mean', accuracy_std='std', num_permutations='count')
    )

    # Replace NaNs that can appear if only one permutation exists (std undefined)
    form_type_stats['accuracy_std'] = form_type_stats['accuracy_std'].fillna(0.0)

    # Ordering & categoricals
    form_type_stats['question_type'] = pd.Categorical(
        form_type_stats['question_type'],
        categories=question_type_order,
        ordered=True
    )
    form_type_stats = form_type_stats.sort_values(['base_form', 'question_type'])
    base_form_order = sorted(form_type_stats['base_form'].unique())
    n_hue = len(question_type_order)

    # Compute standard error
    form_type_stats['se'] = (
        form_type_stats['accuracy_std'] / np.sqrt(form_type_stats['num_permutations'].clip(lower=1))
    ).fillna(0.0)

    # Create plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=form_type_stats,
        x="base_form", y="avg_accuracy", hue="question_type",
        palette="Set2", alpha=0.9,
        order=base_form_order, hue_order=question_type_order,
        errorbar=None  # we'll add custom error bars
    )

    # Add custom error bars
    for xi, bf in enumerate(base_form_order):
        for hi, qt in enumerate(question_type_order):
            sel = form_type_stats[(form_type_stats['base_form'] == bf) &
                                (form_type_stats['question_type'] == qt)]
            if sel.empty:
                continue

            patch_idx = xi * n_hue + hi
            # Guard: some combinations may not be drawn if absent in data
            if patch_idx >= len(ax.patches):
                continue

            bar = ax.patches[patch_idx]
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()

            se = float(sel['se'].iloc[0])

            if se > 0:
                ax.errorbar(
                    x, y,
                    yerr=se,
                    fmt='none', color='black', capsize=3, elinewidth=1
                )

    # Titles and formatting
    ax.set_title('Model Accuracy by Form and Question Type with Permutation Variation', fontsize=16)
    ax.set_xlabel('Form', fontsize=14)
    ax.set_ylabel('Average Accuracy', fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend(title='Question Type', title_fontsize=12, fontsize=10, loc='upper right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add value labels on the bars
    bar_containers = [c for c in ax.containers if hasattr(c, 'patches')]
    for container in bar_containers:
        ax.bar_label(container, fmt='%.2f', fontsize=9)

    plt.tight_layout()
    
    fig = plt.gcf()
    
    # Save figure if requested
    if save:
        if filename is None:
            filename = "accuracy_by_form_and_question_type"
        save_figure(fig=fig, filename=filename)
    
    return fig, ax


class ExperimentAnalyzer:
    """
    A class for analyzing experiment accuracy results.
    
    This class handles:
    - Loading and processing experiment data
    - Calculating accuracy statistics
    - Generating visualizations
    - Producing summary reports
    """
    
    def __init__(self, results_dir: str = "./results"):
        """
        Initialize the experiment analyzer.
        
        Args:
            results_dir: Base directory for results
        """
        self.results_dir = results_dir
        self.experiment_dirs = []
        self.results_df = None
        self.stats = None
        
    def load_experiments(self, show_progress: bool = True) -> pd.DataFrame:
        """
        Load all experiments from the results directory.
        
        Args:
            show_progress: Whether to show a progress bar
            
        Returns:
            DataFrame with processed results
        """
        self.experiment_dirs = find_experiment_dirs(self.results_dir)
        logger.info(f"Found {len(self.experiment_dirs)} experiment directories")
        
        self.results_df = process_experiment_results(self.experiment_dirs, show_progress)
        logger.info(f"Loaded {len(self.results_df)} question results from {len(self.experiment_dirs)} experiments")
        
        return self.results_df
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """
        Calculate accuracy statistics.
        
        Returns:
            Dictionary with accuracy statistics
        """
        if self.results_df is None:
            raise ValueError("No results loaded. Call load_experiments() first.")
        
        self.stats = calculate_accuracy_statistics(self.results_df)
        return self.stats
    
    def generate_summary(self) -> str:
        """
        Generate a text summary of model performance.
        
        Returns:
            Formatted text summary
        """
        if self.stats is None:
            self.calculate_statistics()
        
        return generate_performance_summary(self.stats)
    
    def plot_all(self, save: bool = True) -> Dict[str, Tuple]:
        """
        Generate all plots.
        
        Args:
            save: Whether to save the figures
            
        Returns:
            Dictionary mapping plot names to (figure, axes) tuples
        """
        if self.results_df is None:
            raise ValueError("No results loaded. Call load_experiments() first.")
        
        plots = {}
        
        # Plot accuracy by difficulty
        plots['difficulty'] = plot_accuracy_by_difficulty(
            self.results_df, save=save, filename="accuracy_by_difficulty"
        )
        
        # Plot accuracy by question type and difficulty
        plots['question_type_difficulty'] = plot_accuracy_by_question_type_and_difficulty(
            self.results_df, save=save, filename="accuracy_by_question_type_and_difficulty"
        )
        
        # Plot accuracy by form
        plots['form'] = plot_accuracy_by_form(
            self.results_df, save=save, filename="accuracy_by_form"
        )
        
        # Plot accuracy by form and question type
        plots['form_question_type'] = plot_accuracy_by_form_and_question_type(
            self.results_df, save=save, filename="accuracy_by_form_and_question_type"
        )
        
        return plots
