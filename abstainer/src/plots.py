# abstainer/src/plots.py
"""
Lightweight plotting helpers for quick inspection.

Functions
---------
print_topk_next_token(tok, logits, k=10)
    Text-only dump of the top-k next-token strings, logits, and probabilities.

plot_topk_next_token(tok, logits, k=10, ax=None, title=None)
    Matplotlib bar chart of top-k next-token probabilities.

Notes
-----
- `logits` must be a 1D tensor of shape [vocab] for the *next token*.
- We keep this module minimal and UI-agnostic. No global styling.
"""

from __future__ import annotations

# Type annotations using modern syntax
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

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
