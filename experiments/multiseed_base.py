"""Multi-seed replication of base encoders (HerBERT-base, XLM-R-base) on TwitterEmo."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import expit
from sklearn.metrics import (f1_score, hamming_loss, jaccard_score, accuracy_score,
                             precision_score, recall_score)
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding,
                          EarlyStoppingCallback, set_seed)

from thesis_lib import optimal_thresholds, bootstrap_f1_ci as f1_macro_ci  # evaluate ZOSTAJE inline (7 metryk, inny schemat CSV)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results"
# Checkpoints are throwaway (only the CSV metrics are kept); MULTISEED_HF_OUT lets us
# redirect output_dir to a fast local scratch dir, avoiding NTFS torch.save I/O failures.
HF_OUT = Path(os.environ.get("MULTISEED_HF_OUT", ROOT / "data" / "transformers"))
RESULTS.mkdir(parents=True, exist_ok=True)
HF_OUT.mkdir(parents=True, exist_ok=True)

EMOTIONS = ["radość", "smutek", "zaufanie", "wstręt", "strach", "gniew", "przeczuwanie", "zdziwienie"]
MODELS = ["allegro/herbert-base-cased", "FacebookAI/xlm-roberta-base"]
SEEDS = [42, 43, 44]
OUT_CSV = RESULTS / "multiseed_base.csv"

tw_train = pd.read_csv(PROCESSED / "twitteremo_train.csv").reset_index(drop=True)
tw_val = pd.read_csv(PROCESSED / "twitteremo_val.csv").reset_index(drop=True)
tw_test = pd.read_csv(PROCESSED / "twitteremo_test.csv").reset_index(drop=True)
for df in (tw_train, tw_val, tw_test):
    df["tekst"] = df["tekst"].fillna("")
y_val, y_test = tw_val[EMOTIONS].values, tw_test[EMOTIONS].values

_pos = tw_train[EMOTIONS].values.sum(axis=0)
_neg = len(tw_train) - _pos
POS_WEIGHT = torch.tensor(np.clip(_neg / np.maximum(_pos, 1), 1.0, 10.0), dtype=torch.float32)


def evaluate(yt: np.ndarray, yp: np.ndarray) -> dict:
    """Standard multi-label metrics."""
    return {"f1_macro": f1_score(yt, yp, average="macro", zero_division=0),
            "f1_micro": f1_score(yt, yp, average="micro", zero_division=0),
            "precision_macro": precision_score(yt, yp, average="macro", zero_division=0),
            "recall_macro": recall_score(yt, yp, average="macro", zero_division=0),
            "hamming_loss": hamming_loss(yt, yp),
            "jaccard_macro": jaccard_score(yt, yp, average="macro", zero_division=0),
            "subset_accuracy": accuracy_score(yt, yp)}


# find_optimal_thresholds: wrapper na thesis_lib.optimal_thresholds (3. arg ignorowany, siatka 0.05/0.01 = oryginał).
# f1_macro_ci = thesis_lib.bootstrap_f1_ci (import wyżej). evaluate() ZOSTAJE inline — ma 7 metryk (bez f1_weighted),
# inny schemat niż thesis_lib.evaluate; podmiana zmieniłaby kolumny multiseed_base.csv (append-mode).
def find_optimal_thresholds(yt, yp, labels=None):
    return optimal_thresholds(yt, yp)


class WeightedTrainer(Trainer):
    """Trainer with weighted BCE (pos_weight) instead of plain BCEWithLogitsLoss."""

    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = F.binary_cross_entropy_with_logits(
            logits, labels.float(),
            pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None)
        return (loss, outputs) if return_outputs else loss


def train_one(model_name: str, seed: int, epochs: int = 8, batch_size: int = 8,
              lr: float = 2e-5, max_len: int = 128, patience: int = 2) -> dict:
    """Full fine-tune of a base encoder for one seed; returns test metrics dict."""
    set_seed(seed)
    short = model_name.split("/")[-1]
    label = f"{short}-s{seed}"
    print(f"\n=== {label} (full FT, epochs<={epochs}, early stop) ===", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)

    def to_ds(df: pd.DataFrame) -> Dataset:
        d = Dataset.from_dict({"text": df["tekst"].tolist(),
                               "labels": df[EMOTIONS].values.astype("float32").tolist()})
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=max_len),
                     batched=True, remove_columns=["text"])

    ds_train, ds_val, ds_test = to_ds(tw_train), to_ds(tw_val), to_ds(tw_test)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(EMOTIONS), problem_type="multi_label_classification")

    args = TrainingArguments(
        output_dir=str(HF_OUT / label),
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=32,
        gradient_accumulation_steps=2, gradient_checkpointing=True,
        num_train_epochs=epochs, learning_rate=lr, warmup_ratio=0.1, weight_decay=0.01,
        fp16=torch.cuda.is_available(), logging_steps=50, report_to="none", seed=seed)

    def compute_metrics(p):
        pred = (expit(p.predictions) >= 0.5).astype(int)
        return {"f1_macro": f1_score(p.label_ids.astype(int), pred, average="macro", zero_division=0)}

    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
        pos_weight=POS_WEIGHT, callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)])
    trainer.train()

    p_val = expit(trainer.predict(ds_val).predictions)
    p_test = expit(trainer.predict(ds_test).predictions)
    thr = find_optimal_thresholds(y_val, p_val, EMOTIONS)
    pred_test = (p_test >= thr).astype(int)
    m = evaluate(y_test, pred_test)
    _, lo, hi = f1_macro_ci(y_test, pred_test)
    m.update({"model": short, "seed": seed, "ci_low": round(lo, 3), "ci_high": round(hi, 3),
              "minutes": round((time.time() - t0) / 60, 1)})
    print(f"  {label}: F1-Macro={m['f1_macro']:.3f} [{lo:.3f}, {hi:.3f}]  ({m['minutes']} min)", flush=True)

    del trainer, model
    torch.cuda.empty_cache()
    return m


def already_done() -> set[tuple[str, int]]:
    """Read existing CSV so re-launches skip completed (model, seed) pairs."""
    if not OUT_CSV.exists():
        return set()
    df = pd.read_csv(OUT_CSV)
    return set(zip(df["model"], df["seed"]))


def main() -> None:
    done = already_done()
    for model_name in MODELS:
        short = model_name.split("/")[-1]
        for seed in SEEDS:
            if (short, seed) in done:
                print(f"skip {short}-s{seed} (already in CSV)", flush=True)
                continue
            m = train_one(model_name, seed)
            row = pd.DataFrame([m])
            row.to_csv(OUT_CSV, mode="a", header=not OUT_CSV.exists(), index=False)

    df = pd.read_csv(OUT_CSV)
    print("\n=== PODSUMOWANIE (mean +/- std F1-Macro) ===", flush=True)
    for short, g in df.groupby("model"):
        vals = g["f1_macro"].values
        print(f"{short}: {vals.mean():.3f} +/- {vals.std(ddof=1):.3f}  "
              f"(n={len(vals)}, zakres {vals.min():.3f}-{vals.max():.3f}, seedy {sorted(g['seed'])})",
              flush=True)


if __name__ == "__main__":
    main()
