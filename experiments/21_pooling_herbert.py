"""data pooling on a transformer.

Run from experiments/:  THESIS_HF_OUT=/var/tmp/thesis_hf ../.venv/bin/python 21_pooling_herbert.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import expit
from datasets import Dataset
from sklearn.metrics import f1_score, recall_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding,
                          EarlyStoppingCallback, set_seed)

from thesis_lib import EMOTIONS, optimal_thresholds, bootstrap_f1_ci

RANDOM_STATE = 42
RARE = ["strach", "zaufanie", "smutek"]
FREQUENT = [e for e in EMOTIONS if e not in RARE]
MODEL_NAME = "allegro/herbert-base-cased"

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results"
HF_OUT = Path(os.environ.get("THESIS_HF_OUT", "/var/tmp/thesis_hf"))
HF_OUT.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULTS / "pooling_herbert.csv"


def load(prefix: str, text_col: str) -> dict[str, pd.DataFrame]:
    out = {}
    for sp in ("train", "val", "test"):
        df = pd.read_csv(PROCESSED / f"{prefix}_{sp}.csv").reset_index(drop=True)
        df["__text"] = df[text_col].fillna("")
        out[sp] = df
    return out


TW = load("twitteremo", "tekst")
CE = load("clarin_emo", "tekst")
GO = load("go_emotions", "text_pl")
y_val, y_test = TW["val"][EMOTIONS].values, TW["test"][EMOTIONS].values


def pool(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Concatenate __text + emotion labels from multiple frames."""
    return pd.concat([d[["__text"] + EMOTIONS] for d in dfs], ignore_index=True)


rare_ce = CE["train"][CE["train"][RARE].sum(1) > 0]
rare_go = GO["train"][GO["train"][RARE].sum(1) > 0]

VARIANTS = {
    "TW+CE": pool(TW["train"], CE["train"]),
    "TW+rare(CE,GO)": pool(TW["train"], rare_ce, rare_go),
}


class WeightedTrainer(Trainer):
    """Trainer with weighted BCE (pos_weight) instead of plain BCEWithLogitsLoss."""

    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.binary_cross_entropy_with_logits(
            outputs.logits.float(), labels.float(),
            pos_weight=self.pos_weight.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss


def train_variant(name: str, train_df: pd.DataFrame, epochs: int = 8,
                  batch_size: int = 8, lr: float = 2e-5, patience: int = 2) -> dict:
    """Fine-tune HerBERT-base on the pooled train; eval on TW test (recipe = multiseed_base)."""
    set_seed(RANDOM_STATE)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def to_ds(texts: list[str], y: np.ndarray) -> Dataset:
        d = Dataset.from_dict({"text": texts, "labels": y.astype("float32").tolist()})
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=128),
                     batched=True, remove_columns=["text"])

    ds_train = to_ds(train_df["__text"].tolist(), train_df[EMOTIONS].values)
    ds_val = to_ds(TW["val"]["__text"].tolist(), y_val)
    ds_test = to_ds(TW["test"]["__text"].tolist(), y_test)

    pos = train_df[EMOTIONS].values.sum(axis=0)
    neg = len(train_df) - pos
    pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 10.0), dtype=torch.float32)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(EMOTIONS), problem_type="multi_label_classification")
    args = TrainingArguments(
        output_dir=str(HF_OUT / f"pool21_{name.replace('(', '_').replace(')', '').replace(',', '_')}"),
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=32,
        gradient_accumulation_steps=2, gradient_checkpointing=True,
        num_train_epochs=epochs, learning_rate=lr, warmup_ratio=0.1, weight_decay=0.01,
        fp16=torch.cuda.is_available(), logging_steps=100, report_to="none", seed=RANDOM_STATE)

    def compute_metrics(p):
        pred = (expit(p.predictions) >= 0.5).astype(int)
        return {"f1_macro": f1_score(p.label_ids.astype(int), pred, average="macro", zero_division=0)}

    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
        pos_weight=pw, callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)])
    trainer.train()

    p_val = expit(trainer.predict(ds_val).predictions)
    p_test = expit(trainer.predict(ds_test).predictions)
    thr = optimal_thresholds(y_val, p_val)
    pred = (p_test >= thr).astype(int)
    base, lo, hi = bootstrap_f1_ci(y_test, pred)

    row = {"wariant": name, "n_train": len(train_df),
           "f1_macro": base, "ci_low": round(lo, 3), "ci_high": round(hi, 3),
           "f1_micro": f1_score(y_test, pred, average="micro", zero_division=0),
           "minutes": round((time.time() - t0) / 60, 1)}
    for c in RARE:
        i = EMOTIONS.index(c)
        row[f"recall_{c}"] = recall_score(y_test[:, i], pred[:, i], zero_division=0)
        row[f"f1_{c}"] = f1_score(y_test[:, i], pred[:, i], zero_division=0)
    row["f1_rare_avg"] = float(np.mean([row[f"f1_{c}"] for c in RARE]))
    row["f1_frequent_avg"] = float(np.mean(
        [f1_score(y_test[:, EMOTIONS.index(e)], pred[:, EMOTIONS.index(e)], zero_division=0)
         for e in FREQUENT]))

    del trainer, model
    torch.cuda.empty_cache()
    return row


def main() -> None:
    done = set(pd.read_csv(OUT_CSV)["wariant"]) if OUT_CSV.exists() else set()
    for name, df in VARIANTS.items():
        if name in done:
            print(f"skip {name} (already in CSV)", flush=True)
            continue
        print(f"=== HerBERT pooling: {name} (n_train={len(df)}) ===", flush=True)
        row = train_variant(name, df)
        pd.DataFrame([row]).to_csv(OUT_CSV, mode="a", header=not OUT_CSV.exists(), index=False)
        print(f"  -> {name}: F1-Macro={row['f1_macro']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}] "
              f"rare_avg={row['f1_rare_avg']:.3f}  ({row['minutes']} min)", flush=True)

    print("\n=== pooling_herbert.csv ===", flush=True)
    print(pd.read_csv(OUT_CSV).round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
