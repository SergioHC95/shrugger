# abstainer/src/baselines.py
import torch


def temperature_scale(logits: torch.Tensor, T: float) -> torch.Tensor:
    """Return logits / T. T>1 flattens, T<1 sharpens."""
    return logits / max(T, 1e-6)


def norm_scale(logits: torch.Tensor, s: float) -> torch.Tensor:
    """Scale vector norm to s * ||logits|| (centered)."""
    mu = logits.mean()
    v = logits - mu
    norm = torch.linalg.vector_norm(v) + 1e-12
    return mu + (s * norm) * (v / norm)
