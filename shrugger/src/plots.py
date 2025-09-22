# shrugger/src/plots.py
"""
Lightweight plotting helpers for quick inspection.

Functions
---------
print_topk_next_token(tok, logits, k=10)
    Text-only dump of the top-k next-token strings, logits, and probabilities.

plot_topk_next_token(tok, logits, k=10, ax=None, title=None)
    Matplotlib bar chart of top-k next-token probabilities.

plot_projections(eval_results, title=None)
    Plot histograms of projections onto the abstention direction.

plot_layer_performance(evaluations, best_layer=None)
    Plot AUC and Cohen's d metrics across layers with dual y-axes.

plot_roc_curve(y_true, y_score)
    Plot ROC curve with AUC score.

save_figure(fig, filename, directory=None)
    Save a figure to a file with proper formatting.

"""

from __future__ import annotations

# Type annotations using modern syntax
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

try:
    import seaborn as sns
except ImportError:
    sns = None  # Make seaborn optional
try:
    from sklearn.metrics import roc_curve
except ImportError:
    roc_curve = None  # Make sklearn optional

# ------------------------------- core utils -------------------------------


def _topk_from_logits(
    logits: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (top_probs, top_logits, top_indices) for next-token distribution.
    """
    if logits.dim() != 1:
        raise ValueError(f"logits must be 1D [vocab], got shape {tuple(logits.shape)}")
    probs = F.softmax(logits, dim=-1)
    k = min(int(k), logits.numel())
    top_probs, top_idx = torch.topk(probs, k)
    top_logits = logits[top_idx]
    return top_probs, top_logits, top_idx


def _decode_tokens(tok, idx: torch.Tensor) -> list[str]:
    """
    Decode a 1D tensor of token IDs into readable strings.
    Decodes each token separately to avoid merges across tokens.
    """
    toks = []
    for i in idx.tolist():
        try:
            toks.append(tok.decode([i]))
        except Exception:
            toks.append(f"<{i}>")
    return toks


# ------------------------------- public API -------------------------------


def print_topk_next_token(tok, logits: torch.Tensor, k: int = 10) -> None:
    """
    Text-only dump of top-k next-token strings, logits, and probabilities.
    """
    top_probs, top_logits, top_idx = _topk_from_logits(logits, k)
    tok_strs = _decode_tokens(tok, top_idx)

    # Pretty aligned print
    widest = max(len(s) for s in tok_strs) if tok_strs else 1
    print("Top-k next-token distribution")
    for s, lg, pr in zip(tok_strs, top_logits.tolist(), top_probs.tolist()):
        # Replace newlines to keep one-line rows
        s_disp = s.replace("\n", "\\n")
        print(f"{s_disp:<{widest}}  logit={lg:>8.4f}  prob={pr:>8.4f}")


def plot_topk_next_token(
    tok,
    logits: torch.Tensor,
    k: int = 10,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    Bar chart of the top-k next-token probabilities.

    Parameters
    ----------
    tok : PreTrainedTokenizer
        The tokenizer used to decode token IDs.
    logits : torch.Tensor
        1D next-token logits [vocab].
    k : int
        Number of top tokens to display.
    ax : plt.Axes, optional
        Existing axes. If None, a new figure/axes is created.
    title : str, optional
        Title for the plot.

    Returns
    -------
    ax : plt.Axes
        The axes containing the bar chart.
    """
    top_probs, top_logits, top_idx = _topk_from_logits(logits, k)
    tok_strs = _decode_tokens(tok, top_idx)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))

    # Horizontal bar chart, most probable at the top
    y = list(range(len(tok_strs)))[::-1]
    probs = top_probs.tolist()[::-1]
    labels = tok_strs[::-1]

    ax.barh(y, probs)
    ax.set_yticks(y)
    # Replace newlines to keep labels on one line
    ax.set_yticklabels([s.replace("\n", "\\n") for s in labels])
    ax.set_xlabel("Probability")
    ax.set_xlim(0, 1)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.8, alpha=0.6)
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Top-k next-token probabilities")

    # Tight layout without assuming figure context
    try:
        ax.figure.tight_layout()
    except Exception:
        pass

    return ax


def save_figure(
    fig=None, filename=None, directory=None, dpi=300, bbox_inches="tight", format="png"
):
    """
    Save a figure to a file with proper formatting.

    Parameters
    ----------
    fig : plt.Figure, optional
        The figure to save. If None, uses the current figure.
    filename : str, optional
        The filename to save to. If None, uses a timestamp.
    directory : str or Path, optional
        The directory to save to. If None, uses './outputs/figures'.
    dpi : int, optional
        The resolution in dots per inch.
    bbox_inches : str, optional
        The bounding box of the figure.
    format : str, optional
        The file format to save as.

    Returns
    -------
    path : Path
        The path to the saved figure.
    """
    import os
    from datetime import datetime
    from pathlib import Path

    # Get the current figure if none provided
    if fig is None:
        fig = plt.gcf()

    # Create a default filename if none provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"figure_{timestamp}"

    # Add extension if not present
    if not filename.lower().endswith(f".{format}"):
        filename = f"{filename}.{format}"

    # Create the directory if it doesn't exist
    if directory is None:
        # Get path relative to the module location
        import os.path

        module_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        project_root = (
            module_dir.parent.parent
        )  # Go up two levels from shrugger/src to project root
        directory = project_root / "outputs" / "figures"
    else:
        directory = Path(directory)

    os.makedirs(directory, exist_ok=True)

    # Save the figure
    filepath = directory / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, format=format)
    print(f"Saved figure to {filepath}")

    return filepath


def plot_projections(
    eval_results,
    title="Projections onto Abstention Direction",
    ax=None,
    use_seaborn=True,
    save=True,
    filename=None,
):
    """
    Plot histograms of projections onto the abstention direction.

    Parameters
    ----------
    eval_results : dict
        Dictionary containing evaluation results with keys:
        - 'proj_pos': Projections for positive class
        - 'proj_neg': Projections for negative class
        - 'mean_pos': Mean projection for positive class
        - 'mean_neg': Mean projection for negative class
        - 'auc': AUC score
        - 'cohen_d': Cohen's d effect size
    title : str, optional
        Title for the plot.
    ax : plt.Axes, optional
        Existing axes. If None, a new figure/axes is created.
    use_seaborn : bool, optional
        Whether to use seaborn for plotting histograms (if available).
    save : bool, optional
        Whether to save the figure to disk.
    filename : str, optional
        Filename to use when saving. If None, a default name will be generated.

    Returns
    -------
    ax : plt.Axes or Path
        The axes containing the histogram plot, or the path to the saved figure if save=True.
    """
    proj_pos = eval_results["proj_pos"]
    proj_neg = eval_results["proj_neg"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Plot histograms
    if use_seaborn and sns is not None:
        # Use seaborn for nicer histograms if available
        sns.histplot(
            proj_neg,
            bins=30,
            alpha=0.7,
            label="Low CA (Non-abstention)",
            color="salmon",
            ax=ax,
        )
        sns.histplot(
            proj_pos,
            bins=30,
            alpha=0.7,
            label="High CA (Abstention)",
            color="skyblue",
            ax=ax,
        )
    else:
        # Fall back to matplotlib
        ax.hist(
            proj_pos, bins=30, alpha=0.7, label="High CA (Abstention)", color="skyblue"
        )
        ax.hist(
            proj_neg,
            bins=30,
            alpha=0.7,
            label="Low CA (Non-abstention)",
            color="salmon",
        )

    # Add vertical lines for means
    ax.axvline(
        eval_results["mean_neg"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Mean Low CA: {eval_results["mean_neg"]:.2f}',
    )
    ax.axvline(
        eval_results["mean_pos"],
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f'Mean High CA: {eval_results["mean_pos"]:.2f}',
    )

    # Add labels and title
    ax.set_xlabel("Projection Value", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.set_title(title, fontsize=16)

    # Add metrics to the plot
    ax.text(
        0.02,
        0.95,
        f"AUC: {eval_results['auc']:.4f}",
        transform=ax.transAxes,
        fontsize=12,
    )
    ax.text(
        0.02,
        0.90,
        f"Cohen's d: {eval_results['cohen_d']:.4f}",
        transform=ax.transAxes,
        fontsize=12,
    )

    # Add legend and grid
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    # Tight layout without assuming figure context
    try:
        ax.figure.tight_layout()
    except Exception:
        pass

    # Save the figure if requested
    if save:
        if filename is None:
            # Generate a default filename based on the title
            filename = f"projections_{title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(':', '')}"
        return save_figure(fig=ax.figure, filename=filename)

    return ax


def plot_layer_performance(
    evaluations,
    best_layer=None,
    figsize=(14, 7),
    title="Classification Performance across Layers",
    save=True,
    filename=None,
):
    """
    Plot AUC and Cohen's d metrics across layers with dual y-axes.

    Parameters
    ----------
    evaluations : dict
        Dictionary mapping layer indices to evaluation results dictionaries,
        each containing at least 'auc' and 'cohen_d' keys.
    best_layer : int, optional
        Layer index to highlight as the best performing layer.
    figsize : tuple, optional
        Figure size as (width, height) in inches.
    title : str, optional
        Title for the plot.
    save : bool, optional
        Whether to save the figure to disk.
    filename : str, optional
        Filename to use when saving. If None, a default name will be generated.

    Returns
    -------
    fig : plt.Figure or Path
        The figure containing the plot, or the path to the saved figure if save=True.
    (ax1, ax2) : tuple
        The primary and secondary axes objects (only returned if save=False).
    """
    # Sort layers by number
    layers = sorted(evaluations.keys())
    aucs = [evaluations[layer]["auc"] for layer in layers]
    cohens_d = [evaluations[layer]["cohen_d"] for layer in layers]

    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # Plot AUC on the first axis
    line1 = ax1.plot(
        layers, aucs, marker="o", linestyle="-", linewidth=2, color="blue", label="AUC"
    )
    ax1.set_xlabel("Layer", fontsize=14)
    ax1.set_ylabel("AUC", fontsize=14, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Plot Cohen's d on the second axis
    line2 = ax2.plot(
        layers,
        cohens_d,
        marker="s",
        linestyle="--",
        linewidth=2,
        color="green",
        label="Cohen's d",
    )
    ax2.set_ylabel("Cohen's d", fontsize=14, color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    # Add title
    plt.title(title, fontsize=16)

    # Add grid
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Highlight the best layer if provided
    if best_layer is not None and best_layer in evaluations:
        ax1.scatter(
            [best_layer], [evaluations[best_layer]["auc"]], color="red", s=100, zorder=5
        )
        ax1.text(
            best_layer,
            evaluations[best_layer]["auc"],
            f"  Best: Layer {best_layer}",
            fontsize=12,
            verticalalignment="center",
        )

    # Add a combined legend
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper center", fontsize=12)

    # Tight layout
    try:
        fig.tight_layout()
    except Exception:
        pass

    # Save the figure if requested
    if save:
        if filename is None:
            # Generate a default filename based on the title
            filename = f"layer_performance_{title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(':', '')}"
        return save_figure(fig=fig, filename=filename)

    return fig, (ax1, ax2)


def plot_roc_curve(
    y_true,
    y_score,
    figsize=(8, 8),
    title="ROC Curve",
    ax=None,
    save=True,
    filename=None,
):
    """
    Plot ROC curve with AUC score.

    Parameters
    ----------
    y_true : array-like
        Binary class labels (1 for positive, 0 for negative).
    y_score : array-like
        Scores/probabilities for the positive class.
    figsize : tuple, optional
        Figure size as (width, height) in inches.
    title : str, optional
        Title for the plot.
    ax : plt.Axes, optional
        Existing axes. If None, a new figure/axes is created.
    save : bool, optional
        Whether to save the figure to disk.
    filename : str, optional
        Filename to use when saving. If None, a default name will be generated.

    Returns
    -------
    ax : plt.Axes or Path
        The axes containing the ROC curve, or the path to the saved figure if save=True.
    """
    if roc_curve is None:
        raise ImportError(
            "sklearn.metrics is required for ROC curve plotting. "
            "Please install scikit-learn."
        )

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # Calculate AUC
    try:
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(y_true, y_score)
    except ImportError:
        # Calculate AUC manually using trapezoidal rule if sklearn not available
        auc = np.trapz(tpr, fpr)

    # Create plot if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot ROC curve
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Tight layout without assuming figure context
    try:
        ax.figure.tight_layout()
    except Exception:
        pass

    # Save the figure if requested
    if save:
        if filename is None:
            # Generate a default filename based on the title
            filename = f"roc_curve_{title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(':', '')}"
        return save_figure(fig=ax.figure, filename=filename)

    return ax
