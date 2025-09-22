# /content/drive/MyDrive/MATS-Project/run.py
import json
import os
import sys

from abstainer import (
    build_quantitative_prompt,
    decision_rule,
    load_model,
    next_token_logits,
    probs_from_logits,
    tokenizer_token_ids,
)


def main():
    ROOT = os.path.dirname(os.path.abspath(__file__))
    CONFIG_FILE = os.path.join(ROOT, "..", "config.json")
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: config.json file not found at {CONFIG_FILE}")
        print("Please create a config.json file with your Hugging Face token.")
        sys.exit(1)

    cfg = json.load(open(CONFIG_FILE))
    if "hf_token" not in cfg:
        print("Error: 'hf_token' not found in config.json")
        print("Please add your Hugging Face token to config.json")
        sys.exit(1)

    os.environ["HF_TOKEN"] = cfg["hf_token"]
    print(
        f"Loaded config: model_id={cfg.get('model_id')}, dtype={cfg.get('dtype','auto')}"
    )

    # Load model (handles CPU/GPU policy internally)
    tok, mdl = load_model(cfg["model_id"], dtype=cfg.get("dtype", "auto"))
    p = next(mdl.parameters())
    print(f"Model loaded on device={p.device}, dtype={p.dtype}")

    # Example prompt
    t = 0.5
    prompt = build_quantitative_prompt(
        "The Pacific Ocean is larger than all of Earth’s land area combined.",
        t,
        form="A_X",
    )
    logits = next_token_logits(tok, mdl, prompt)
    probs = probs_from_logits(logits)

    # Pull pT, pF directly
    ids = tokenizer_token_ids(tok, ["T", "F"])
    pT = (
        probs[ids["T"][0]].item()
        if len(ids["T"]) == 1
        else sum(probs[i].item() for i in ids["T"])
    )
    pF = (
        probs[ids["F"][0]].item()
        if len(ids["F"]) == 1
        else sum(probs[i].item() for i in ids["F"])
    )

    decision, pmax = decision_rule(pT, pF, t)
    print(
        f"Gemma3={cfg['model_id']}  pT={pT:.4f} pF={pF:.4f}  decision={decision}  pmax={pmax:.4f}"
    )


if __name__ == "__main__":
    main()
