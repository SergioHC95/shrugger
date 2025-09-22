from typing import Optional

import torch

from .model import next_token_logits, probs_from_logits
from .prompts import build_likert_prompt
from .utils import (
    get_default_likert_labels,
    get_likert_mapping,
    normalize_likert_output,
)


def run_likert_probe(
    tokenizer,
    model,
    question: str,
    form: str = "V0",
    labels: Optional[list[str]] = None,
):
    """
    Run the model on a Likert scale prompt and extract logits + probabilities.

    Args:
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        question (str): question to ask
        form (str): prompt template to use (default = "V0")
        labels (Optional[list[str]]): Labels to use in order
                                    [definitely_yes, probably_yes, not_sure, probably_no, definitely_no]

    Returns:
        dict with:
            - form:              prompt form used
            - labels:            labels used in the prompt
            - prompt:            input prompt string
            - pred_label:        the raw predicted label from the model
            - canonical_label:   the predicted label in canonical format (YY, Y, A, N, NN)
            - token_ids:         mapping of labels to token IDs
            - canonical_token_ids: { "YY": id, "Y": id, "A": id, "N": id, "NN": id }
            - logits:            mapping of labels to logit values
            - canonical_logits:  { "YY": float, "Y": float, "A": float, "N": float, "NN": float }
            - probs:             mapping of labels to probability values over full vocab
            - canonical_probs:   { "YY": float, "Y": float, "A": float, "N": float, "NN": float }
            - probs_norm:        mapping of labels to normalized probability values
            - canonical_probs_norm: { "YY": float, "Y": float, "A": float, "N": float, "NN": float }
            - is_valid:          boolean flag indicating if the output was successfully normalized
    """
    # Use default labels if none provided
    if labels is None:
        labels = get_default_likert_labels()

    # Build the prompt with the specified labels
    prompt = build_likert_prompt(question, form=form, labels=labels)

    # Get logits for next token
    with torch.no_grad():
        logits = next_token_logits(tokenizer, model, prompt)

    probs = probs_from_logits(logits)

    # Map labels to canonical format
    label_mapping = get_likert_mapping(labels)

    # Map tokens to IDs
    token_map = {
        opt: tokenizer.encode(opt, add_special_tokens=False)[0] for opt in labels
    }

    # Collect values using the provided labels
    logits_sel = {opt: logits[token_id].item() for opt, token_id in token_map.items()}
    probs_sel = {opt: probs[token_id].item() for opt, token_id in token_map.items()}

    # Normalize over the provided labels
    total = sum(probs_sel.values())
    probs_norm = {opt: prob / total for opt, prob in probs_sel.items()}

    # Decode top-1 output token
    top_id = torch.argmax(logits).item()
    pred_label = tokenizer.decode([top_id])

    # Normalize the output to canonical format (YY, Y, A, N, NN)
    canonical_label, is_valid = normalize_likert_output(pred_label, labels=labels)

    # Convert results to canonical format
    canonical_token_map = {
        label_mapping[opt]: token_id for opt, token_id in token_map.items()
    }
    canonical_logits = {label_mapping[opt]: val for opt, val in logits_sel.items()}
    canonical_probs = {label_mapping[opt]: val for opt, val in probs_sel.items()}
    canonical_probs_norm = {label_mapping[opt]: val for opt, val in probs_norm.items()}

    return {
        "form": form,
        "labels": labels,
        "prompt": prompt,
        "pred_label": pred_label,
        "canonical_label": canonical_label,
        "token_ids": token_map,
        "canonical_token_ids": canonical_token_map,
        "logits": logits_sel,
        "canonical_logits": canonical_logits,
        "probs": probs_sel,
        "canonical_probs": canonical_probs,
        "probs_norm": probs_norm,
        "canonical_probs_norm": canonical_probs_norm,
        "is_valid": is_valid,
    }


def run_yesno_probe(
    tokenizer, model, prompt: str, X_token: str = "X", detailed: bool = False
):
    """
    Run the model on a Yes/No/X prompt and extract logits + probabilities.

    Args:
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        prompt (str): full prompt string
        X_token (str): token to use for the abstain option (default = "X")

    Returns:
        dict with:
            - prompt:    input prompt string
            - output:    decoded top-1 token string
            - logits:    { "Y": float, "N": float, "X": float }
            - probs:     { "Y": float, "N": float, "X": float }       # over full vocab
            - probs_norm:{ "Y": float, "N": float, "X": float }       # renormalized over {Y,N,X}
    """
    # Get logits for next token
    with torch.no_grad():
        logits = next_token_logits(tokenizer, model, prompt)

    probs = probs_from_logits(logits)

    # Map tokens to IDs
    token_map = {
        "Y": tokenizer.encode("Y", add_special_tokens=False)[0],
        "N": tokenizer.encode("N", add_special_tokens=False)[0],
        "X": tokenizer.encode(X_token, add_special_tokens=False)[0],
    }

    # Collect values
    logits_sel = {k: logits[v].item() for k, v in token_map.items()}
    probs_sel = {k: probs[v].item() for k, v in token_map.items()}

    # Normalize over Y/N/X
    total = sum(probs_sel.values())
    probs_norm = {k: v / total for k, v in probs_sel.items()}

    # Decode top-1 output token
    top_id = torch.argmax(logits).item()
    output_token = tokenizer.decode([top_id])

    if detailed:
        return {
            "prompt": prompt,
            "output": output_token,
            "logits": logits_sel,
            "probs": probs_sel,
            "probs_norm": probs_norm,
        }
    else:
        return probs_sel
