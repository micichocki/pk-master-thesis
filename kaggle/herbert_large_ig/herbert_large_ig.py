"""Kernel inputs:
  - dataset  miczimici/pl-emotion-processed   (test split)
  - kernel   miczimici/herbert-large-full-emotion  (fine-tuned checkpoint, via kernel_sources)
Output: /kaggle/working/herbert_ig.json  (tokens + attributions per sentence/emotion)
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

EMOTIONS = ["radość", "smutek", "zaufanie", "wstręt", "strach", "gniew", "przeczuwanie", "zdziwienie"]

# same test-row indices as the classical part (experiments/18_interpretability.py)
SENTENCES = {
    646: ["radość"],
    1290: ["wstręt"],
    556: ["strach"],
    1362: ["przeczuwanie"],
    1207: ["wstręt", "gniew"],   # sarcastic
    985: ["wstręt", "gniew"],    # multi-label
}

# ------------------------------------------------------------------ locate checkpoint
print("INPUT TREE:")
for p in sorted(glob.glob("/kaggle/input/*")):
    print("  ", p)
configs = glob.glob("/kaggle/input/**/config.json", recursive=True)
ckpt = None
for c in configs:
    d = os.path.dirname(c)
    if "checkpoint" in c and os.path.exists(os.path.join(d, "model.safetensors")):
        ckpt = d
        break
if ckpt is None:  # fallback: any dir with weights
    for c in configs:
        d = os.path.dirname(c)
        if os.path.exists(os.path.join(d, "model.safetensors")):
            ckpt = d
            break
assert ckpt is not None, f"checkpoint not found; configs={configs}"
print("CKPT:", ckpt)

os.system("pip install -q captum")
from captum.attr import LayerIntegratedGradients  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device).eval()
print("loaded:", model.config.model_type, "num_labels=", model.config.num_labels, "device=", device)

csvs = glob.glob("/kaggle/input/**/twitteremo_test.csv", recursive=True)
assert csvs, "twitteremo_test.csv not found under /kaggle/input"
print("TEST CSV:", csvs[0])
test = pd.read_csv(csvs[0])
test["tekst"] = test["tekst"].fillna("")


def forward_logits(input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask).logits


lig = LayerIntegratedGradients(forward_logits, model.get_input_embeddings())


def attribute(text: str, e_idx: int):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    special = tok.get_special_tokens_mask(input_ids[0].tolist(), already_has_special_tokens=True)
    ref = input_ids.clone()
    for i, s in enumerate(special):
        if s == 0:
            ref[0, i] = tok.pad_token_id
    atts, delta = lig.attribute(
        inputs=input_ids, baselines=ref, additional_forward_args=(attn,),
        target=e_idx, n_steps=200, internal_batch_size=16, return_convergence_delta=True,
    )
    atts = atts.sum(dim=-1).squeeze(0)                 # sum over hidden dim -> per token
    norm = torch.norm(atts)
    if norm > 0:
        atts = atts / norm
    tokens = tok.convert_ids_to_tokens(input_ids[0])
    return tokens, atts.detach().cpu().tolist(), float(delta)


results = []
for idx, emos in SENTENCES.items():
    text = test["tekst"].iloc[idx]
    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        proba = torch.sigmoid(model(**enc).logits)[0].cpu().numpy()
    for emo in emos:
        e_idx = EMOTIONS.index(emo)
        tokens, atts, delta = attribute(text, e_idx)
        results.append({
            "test_index": int(idx), "emotion": emo, "tekst": text,
            "tokens": tokens, "attributions": atts,
            "proba": float(proba[e_idx]), "conv_delta": delta,
        })
        print(f"idx={idx} {emo}: proba={proba[e_idx]:.3f} n_tok={len(tokens)} delta={delta:.2e}")

with open("/kaggle/working/herbert_ig.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("saved /kaggle/working/herbert_ig.json with", len(results), "records")
