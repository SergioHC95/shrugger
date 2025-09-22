# shrugger/src/parsing.py
from __future__ import annotations

import re
from typing import Literal

Answer = Literal["T", "F", "X", "IDK"]

_REFUSAL_PAT = re.compile(
    r"(cannot|can't|won't|refuse|unsafe|policy|not (?:allowed|able)|"
    r"harmful|jailbreak|content policy|safety (?:policy|reason))",
    re.I,
)


def is_refusal(text: str) -> bool:
    """Heuristic refusal detector (benign safety refusals)."""
    return bool(_REFUSAL_PAT.search(text))


def strict_parse(text: str, allow_idk: bool = True) -> tuple[Answer | None, bool]:
    """
    Return (answer, compliant). compliant=False if output is not in {T,F,(X|IDK)}.
    """
    s = text.strip()
    if allow_idk:
        if s in ("T", "F", "X", "IDK"):
            return s, True
    else:
        if s in ("T", "F"):
            return s, True
    return None, False


def decision_rule(pT: float, pF: float, t: float) -> tuple[str | None, float]:
    """
    Implement Bayes rule under (+1, 0, -t). Answer if p_max > t/(1+t), else abstain.
    Returns (decision, p_max). decision in {"T","F",None}
    """
    pmax = max(pT, pF)
    return (("T" if pT >= pF else "F"), pmax) if pmax > (t / (1 + t)) else (None, pmax)
