# shrugger/src/steer.py
from __future__ import annotations

import numpy as np
import torch


def whitened_mean_diff(
    X_ans: np.ndarray, X_abs: np.ndarray, reg: float = 1e-4
) -> np.ndarray:
    """
    Whitened mean difference: v ∝ Σ^{-1} (μ_ans - μ_abs)
    X_*: [N, d] activations at the answer token for a given layer (float64/float32).
    Returns L2-normalized v (float64).
    """
    mu_a = X_ans.mean(0, keepdims=True)
    mu_b = X_abs.mean(0, keepdims=True)
    Xc = np.vstack([X_ans - mu_a, X_abs - mu_b])
    d = Xc.shape[1]
    cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1) + reg * np.eye(d)
    diff = (mu_a - mu_b).ravel()
    v = np.linalg.solve(cov, diff)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v


# ---------- Causal intervention hook ----------


class SteerHook:
    """
    Add a vector α v to the residual stream at a chosen module (layer) for the *last position only*.
    Usage:
        hook = SteerHook(model, module_name, v, alpha)
        with hook:
            out = model(**inputs)
    You must know the module to hook (e.g., a LayerNorm or residual pre tensor).
    """

    def __init__(self, model, module_name: str, v: np.ndarray, alpha: float):
        self.model = model
        self.module_name = module_name
        self.v = torch.tensor(v, dtype=model.dtype, device=self._device_of_model())
        self.alpha = float(alpha)
        self.handle = None
        self._module = dict(model.named_modules()).get(module_name, None)
        if self._module is None:
            raise ValueError(f"Module {module_name} not found in model.")

    def _device_of_model(self):
        # heuristic: first param device
        return next(self.model.parameters()).device

    def __enter__(self):
        def _hook(module, inp, out):
            # out: [batch, seq, d] or [batch, d] depending on module; we expect [B,S,D]
            if out.dim() == 3:
                out = out.clone()
                out[:, -1, :] = out[:, -1, :] + self.alpha * self.v
                return out
            elif out.dim() == 2:
                out = out.clone()
                out[:, :] = out[:, :] + self.alpha * self.v  # fallback
                return out
            else:
                return out

        self.handle = self._module.register_forward_hook(_hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
