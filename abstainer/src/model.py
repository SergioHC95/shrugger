# abstainer/src/model.py
from __future__ import annotations

import os

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_dtype(policy: str | None) -> torch.dtype:
    """
    Resolve dtype from a user policy string.
    Defaults to bf16 if policy is None.

    Accepts: "bf16"/"bfloat16", "fp16"/"float16", "fp32"/"float32", "auto"
    """
    if policy is None:
        return torch.bfloat16  # default to bf16

    p = policy.lower()
    if p in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if p in {"fp16", "float16"}:
        return torch.float16
    if p in {"fp32", "float32"}:
        return torch.float32

    # "auto" keeps your previous semantics, but still bf16-first
    if p == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.bfloat16  # on CPU, prefer bf16 for memory footprint

    # Unknown string → default to bf16
    return torch.bfloat16


def load_model(model_id: str, dtype: str | None = None, device_map: str | None = None):
    """
    Returns (tokenizer, model) with a bf16-first policy:

      - Default dtype = bf16 everywhere.
      - CUDA:
          * If bf16 supported → bf16
          * Else → warn + fp16
      - CPU:
          * Try bf16 (saves RAM); if load fails for non-OOM reasons, try fp32.
          * If OOM in either → raise with guidance.

    Notes:
      - Uses Transformers' modern kwarg: dtype=...
      - low_cpu_mem_usage=True to reduce peak memory when loading.
    """
    token = os.environ.get("HF_TOKEN", None)
    requested_dtype = _resolve_dtype(dtype)

    # Decide device map (explicit beats implicit)
    if device_map is None:
        device_map = "auto" if torch.cuda.is_available() else "cpu"

    # GPU path
    if torch.cuda.is_available():
        chosen = requested_dtype
        if requested_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            print(
                "[model] Warning: CUDA bf16 not supported on this GPU; falling back to fp16."
            )
            chosen = torch.float16

        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=token)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=chosen,
            device_map=device_map,
            low_cpu_mem_usage=True,
            token=token,
        )
        mdl.eval()
        return tok, mdl

    # CPU path — prefer bf16 to reduce RAM; fallback to fp32 if needed
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=token)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,  # default to bf16 on CPU
            device_map="cpu",
            low_cpu_mem_usage=True,
            token=token,
        )
        mdl.eval()
        return tok, mdl
    except RuntimeError as e:
        msg = str(e).lower()
        if any(x in msg for x in ["out of memory", "oom", "killed"]):
            raise RuntimeError(
                "[model] OOM while loading on CPU with bf16.\n"
                "Options:\n"
                "  • Switch to a GPU runtime (bf16 preferred).\n"
                "  • Use a smaller model (e.g., 'google/gemma-3-2b-it').\n"
                "  • Increase system RAM.\n"
            ) from e
        # Non-OOM issues → try fp32 once
        try:
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=token)
            mdl = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
                token=token,
            )
            mdl.eval()
            print("[model] Loaded in fp32 on CPU as a fallback.")
            return tok, mdl
        except RuntimeError as e2:
            raise RuntimeError(
                "[model] Failed to load on CPU (bf16 then fp32). "
                "Try a smaller model or enable a GPU."
            ) from e2


@torch.no_grad()
def next_token_logits(tok, mdl, prompt: str) -> torch.Tensor:
    """Return logits for the next token at the final position. Shape [vocab]."""
    device = next(mdl.parameters()).device
    ids = tok(prompt, return_tensors="pt")
    ids = {k: v.to(device) for k, v in ids.items()}

    with torch.inference_mode():
        out = mdl(**ids)
        logits = out.logits[0, -1, :]

    if torch.isnan(logits).any():
        raise RuntimeError(
            "[model] NaN logits detected. If on CPU, consider switching to GPU or a smaller model."
        )
    return logits


def tokenizer_token_ids(tok, symbols: list[str]) -> dict[str, list[int]]:
    return {s: tok(s, add_special_tokens=False).input_ids for s in symbols}


def probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)
