"""HerBERT-large — GLOBALNA ważność tokenów z Integrated Gradients."""
import glob
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

EMOTIONS = ["radość", "smutek", "zaufanie", "wstręt", "strach", "gniew", "przeczuwanie", "zdziwienie"]
N_STEPS = 50            # 200 jak w kernelu lokalnym byłoby 4x droższe przy 1435 tekstach
THRESHOLD = 0.5
TIME_BUDGET_S = 9 * 3600   # zapas wobec limitu 12 h; po przekroczeniu zapis i wyjście
SAVE_EVERY = 200

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
if ckpt is None:
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
assert csvs, "twitteremo_test.csv not found"
test = pd.read_csv(csvs[0])
test["tekst"] = test["tekst"].fillna("")
print("tekstów testowych:", len(test))


def forward_logits(input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask).logits


lig = LayerIntegratedGradients(forward_logits, model.get_input_embeddings())

# (emocja, token) -> [liczba wystąpień, suma wkładów, suma modułów wkładów]
acc = defaultdict(lambda: [0, 0.0, 0.0])
texts_per_emotion = defaultdict(int)


def attribute(input_ids, attn, e_idx):
    """Zwraca wkład per token; BEZ normalizacji długością — sumy muszą być porównywalne
    między tekstami, a dzielenie przez normę własną tekstu by to zepsuło."""
    special = tok.get_special_tokens_mask(input_ids[0].tolist(), already_has_special_tokens=True)
    ref = input_ids.clone()
    for i, s in enumerate(special):
        if s == 0:
            ref[0, i] = tok.pad_token_id
    atts = lig.attribute(inputs=input_ids, baselines=ref, additional_forward_args=(attn,),
                         target=e_idx, n_steps=N_STEPS, internal_batch_size=32)
    return atts.sum(dim=-1).squeeze(0).detach().cpu().numpy()


def dump(n_done: int) -> None:
    rows = []
    for (emo, token), (n_occ, s, s_abs) in acc.items():
        rows.append({"emotion": emo, "token": token, "n_occ": n_occ,
                     "sum_attr": s, "sum_abs_attr": s_abs,
                     "n_texts_emotion": texts_per_emotion[emo]})
    df = pd.DataFrame(rows).sort_values(["emotion", "sum_abs_attr"], ascending=[True, False])
    df.to_csv("/kaggle/working/ig_global.csv", index=False)
    print(f"[zapis] {len(df)} wierszy po {n_done} tekstach", flush=True)


t0 = time.time()
for i, text in enumerate(test["tekst"]):
    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        proba = torch.sigmoid(model(**enc).logits)[0].cpu().numpy()
    targets = [j for j in range(len(EMOTIONS)) if proba[j] >= THRESHOLD]
    if targets:
        input_ids, attn = enc["input_ids"], enc["attention_mask"]
        tokens = tok.convert_ids_to_tokens(input_ids[0])
        for e_idx in targets:
            emo = EMOTIONS[e_idx]
            texts_per_emotion[emo] += 1
            for t, a in zip(tokens, attribute(input_ids, attn, e_idx)):
                cell = acc[(emo, t)]
                cell[0] += 1
                cell[1] += float(a)
                cell[2] += abs(float(a))
    if (i + 1) % SAVE_EVERY == 0:
        el = time.time() - t0
        print(f"{i+1}/{len(test)} | {el/60:.1f} min | prognoza {el/(i+1)*len(test)/60:.0f} min", flush=True)
        dump(i + 1)
    if time.time() - t0 > TIME_BUDGET_S:
        print("BUDŻET CZASU WYCZERPANY — zapis częściowy", flush=True)
        break

dump(len(test))
print("gotowe w", (time.time() - t0) / 60, "min")
