"""Bielik-4.5B-v3.0-Instruct — zero-shot & few-shot (in-context learning) na TwitterEmo."""
import os
os.system("pip install -q -U 'bitsandbytes>=0.46.1' accelerate")

import glob
import json
import re
import unicodedata

import numpy as np
import pandas as pd
import torch

EMOTIONS = ["radość", "smutek", "zaufanie", "wstręt", "strach", "gniew", "przeczuwanie", "zdziwienie"]
MODEL_NAME = "speakleash/Bielik-4.5B-v3.0-Instruct"
RANDOM_STATE = 42
N_SHOTS = 5
MAX_NEW_TOKENS = 32
BATCH = 16

# ------------------------------------------------------------------ HF auth (gated)
from huggingface_hub import login
try:
    from kaggle_secrets import UserSecretsClient
    login(UserSecretsClient().get_secret("HF_TOKEN"))
    print("HF zalogowany (sekret)")
except Exception as e:
    raise SystemExit(f"Brak sekretu HF_TOKEN w UI Kaggle: {e}")


def find(pat: str) -> str:
    hits = glob.glob(f"/kaggle/input/**/{pat}", recursive=True)
    assert hits, f"{pat} nie znalezione w /kaggle/input"
    return hits[0]


tw_train = pd.read_csv(find("twitteremo_train.csv")).reset_index(drop=True)
tw_test = pd.read_csv(find("twitteremo_test.csv")).reset_index(drop=True)
for d in (tw_train, tw_test):
    d["tekst"] = d["tekst"].fillna("")
y_test = tw_test[EMOTIONS].values
y_train = tw_train[EMOTIONS].values

# ------------------------------------------------------------------ model (4-bit)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"  # for batched generation
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb, device_map="auto").eval()
print("model zaladowany (4-bit)")

# ------------------------------------------------------------------ prompt building
SYSTEM = (
    "Jesteś precyzyjnym klasyfikatorem emocji w tekstach po polsku. "
    "Dla podanego tekstu wskaż WSZYSTKIE występujące emocje z zamkniętej listy: "
    + ", ".join(EMOTIONS) + ". "
    "Tekst może wyrażać wiele emocji naraz lub żadnej. "
    "Odpowiedz WYŁĄCZNIE nazwami emocji z listy, oddzielonymi przecinkami, bez komentarza. "
    "Jeśli żadna emocja nie występuje, napisz dokładnie: brak."
)


def labels_str(row_vec: np.ndarray) -> str:
    em = [EMOTIONS[i] for i in range(len(EMOTIONS)) if row_vec[i] == 1]
    return ", ".join(em) if em else "brak"


# few-shot: greedily pick examples maximizing emotion coverage (deterministic)
rng = np.random.default_rng(RANDOM_STATE)
order = rng.permutation(len(tw_train))
shots, covered = [], set()
for idx in order:
    pos = {i for i in range(len(EMOTIONS)) if y_train[idx, i] == 1}
    if pos and not pos.issubset(covered):
        shots.append(int(idx))
        covered |= pos
    if len(shots) >= N_SHOTS:
        break
print("few-shot idx:", shots, "pokrycie emocji:", len(covered))


def build_messages(text: str, few_shot: bool):
    msgs = [{"role": "system", "content": SYSTEM}]
    if few_shot:
        for si in shots:
            msgs.append({"role": "user", "content": tw_train["tekst"].iloc[si]})
            msgs.append({"role": "assistant", "content": labels_str(y_train[si])})
    msgs.append({"role": "user", "content": text})
    return msgs


# ------------------------------------------------------------------ parse
def deaccent(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


EMO_KEYS = [(e, deaccent(e).lower()) for e in EMOTIONS]


def parse(out: str) -> np.ndarray:
    low = deaccent(out).lower()
    vec = np.zeros(len(EMOTIONS), dtype=int)
    for i, (_, key) in enumerate(EMO_KEYS):
        if re.search(r"\b" + re.escape(key), low):
            vec[i] = 1
    return vec


# ------------------------------------------------------------------ generate (batched)
def run_condition(few_shot: bool, tag: str):
    prompts = [tok.apply_chat_template(build_messages(t, few_shot), tokenize=False,
                                       add_generation_prompt=True)
               for t in tw_test["tekst"]]
    preds = np.zeros((len(prompts), len(EMOTIONS)), dtype=int)
    samples = []
    for b in range(0, len(prompts), BATCH):
        chunk = prompts[b:b + BATCH]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        outs = tok.batch_decode(new, skip_special_tokens=True)
        for j, o in enumerate(outs):
            preds[b + j] = parse(o)
            if b + j < 12:
                samples.append({"tekst": tw_test["tekst"].iloc[b + j][:120],
                                "output": o.strip()[:120],
                                "pred": labels_str(preds[b + j]),
                                "true": labels_str(y_test[b + j])})
        if b % (BATCH * 10) == 0:
            print(f"[{tag}] {b}/{len(prompts)}")
    return preds, samples


from sklearn.metrics import (f1_score, precision_score, recall_score,
                             jaccard_score, hamming_loss, accuracy_score)


def boot_ci(yt, yp, n=1000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n_s = len(yt)
    vals = [f1_score(yt[idx := rng.integers(0, n_s, n_s)], yp[idx],
                     average="macro", zero_division=0) for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


rows, all_samples = [], {}
all_preds = {}
for few_shot, tag in [(False, "0-shot"), (True, "5-shot")]:
    preds, samples = run_condition(few_shot, tag)
    all_preds[tag] = preds
    all_samples[tag] = samples
    lo, hi = boot_ci(y_test, preds)
    rows.append({
        "warunek": tag,
        "f1_macro": round(f1_score(y_test, preds, average="macro", zero_division=0), 4),
        "ci95": f"[{lo:.3f}; {hi:.3f}]",
        "f1_micro": round(f1_score(y_test, preds, average="micro", zero_division=0), 4),
        "precision_macro": round(precision_score(y_test, preds, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_test, preds, average="macro", zero_division=0), 4),
        "jaccard_macro": round(jaccard_score(y_test, preds, average="macro", zero_division=0), 4),
        "hamming": round(hamming_loss(y_test, preds), 4),
        "subset_acc": round(accuracy_score(y_test, preds), 4),
    })
    print(rows[-1])

pd.DataFrame(rows).to_csv("/kaggle/working/bielik_zeroshot_metrics.csv", index=False)
np.savez("/kaggle/working/bielik_zeroshot_preds.npz",
         **{k: v for k, v in all_preds.items()}, y_test=y_test)
with open("/kaggle/working/bielik_zeroshot_samples.json", "w") as f:
    json.dump({"few_shot_idx": shots, "samples": all_samples}, f, ensure_ascii=False, indent=2)
print("DONE")
