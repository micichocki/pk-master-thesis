"""Re-run XLM-R-base at seeds 43/44, saving probability matrices.

Resumable: a seed whose probability files already exist is skipped.

Usage::

    cd experiments && MULTISEED_HF_OUT=/var/tmp/xlmr_seeds ../.venv/bin/python 38_xlmr_seeds_probas.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from scipy.special import expit
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_lib import bootstrap_f1_ci, evaluate, make_weighted_trainer_cls, optimal_thresholds, pos_weight

EMOTIONS = ["radość", "smutek", "zaufanie", "wstręt", "strach", "gniew", "przeczuwanie", "zdziwienie"]
MODEL = "FacebookAI/xlm-roberta-base"
SEEDS = (43, 44)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED, RESULTS = ROOT / "data/processed", ROOT / "data/results"
PROBAS = RESULTS / "base_encoder_probas"
PROBAS.mkdir(parents=True, exist_ok=True)
HF_OUT = Path(os.environ.get("MULTISEED_HF_OUT", "/var/tmp/xlmr_seeds"))
HF_OUT.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULTS / "xlmr_seeds_probas.csv"

splits = {s: pd.read_csv(PROCESSED / f"twitteremo_{s}.csv").reset_index(drop=True)
          for s in ("train", "val", "test")}
for frame in splits.values():
    frame["tekst"] = frame["tekst"].fillna("")
y = {s: splits[s][EMOTIONS].values for s in splits}

WeightedTrainer = make_weighted_trainer_cls()
POS_WEIGHT = pos_weight(y["train"])


def train_one(seed: int, epochs: int = 8, batch_size: int = 8, lr: float = 2e-5,
              max_len: int = 128, patience: int = 2) -> dict[str, float]:
    """Full fine-tune of XLM-R-base at one seed; saves val/test probability matrices."""
    set_seed(seed)
    run = f"xlm-roberta-base_s{seed}"
    print(f"\n=== {run} ===", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)

    def to_ds(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_dict({"text": df["tekst"].tolist(),
                                "labels": df[EMOTIONS].values.astype("float32").tolist()})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len),
                      batched=True, remove_columns=["text"])

    ds_train, ds_val, ds_test = (to_ds(splits[s]) for s in ("train", "val", "test"))
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(EMOTIONS), problem_type="multi_label_classification")

    args = TrainingArguments(
        output_dir=str(HF_OUT / run),
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=32,
        gradient_accumulation_steps=2, gradient_checkpointing=True,
        num_train_epochs=epochs, learning_rate=lr, warmup_ratio=0.1, weight_decay=0.01,
        fp16=torch.cuda.is_available(), logging_steps=50, report_to="none",
        seed=seed, data_seed=seed)

    def compute_metrics(p) -> dict[str, float]:
        pred = (expit(p.predictions) >= 0.5).astype(int)
        return {"f1_macro": f1_score(p.label_ids.astype(int), pred, average="macro", zero_division=0)}

    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
        pos_weight=POS_WEIGHT, callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)])
    trainer.train()

    p_val = expit(trainer.predict(ds_val).predictions)
    p_test = expit(trainer.predict(ds_test).predictions)
    np.save(PROBAS / f"{run}_val.npy", p_val)
    np.save(PROBAS / f"{run}_test.npy", p_test)

    thr = optimal_thresholds(y["val"], p_val)
    pred_test = (p_test >= thr).astype(int)
    metrics = evaluate(y["test"], pred_test)
    _, lo, hi = bootstrap_f1_ci(y["test"], pred_test)
    metrics.update({"run": run, "model": "xlm-roberta-base", "seed": seed,
                    "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                    "minutes": round((time.time() - t0) / 60, 1)})
    print(f"  {run}: F1-Macro={metrics['f1_macro']:.4f} [{lo:.3f}; {hi:.3f}] "
          f"({metrics['minutes']} min)", flush=True)

    del trainer, model
    torch.cuda.empty_cache()
    return metrics


def append_row(row: dict[str, float]) -> None:
    """Read-concat-write, so a partial CSV stays parseable if the run is interrupted."""
    frame = pd.DataFrame([row])
    if OUT_CSV.exists():
        frame = pd.concat([pd.read_csv(OUT_CSV), frame], ignore_index=True)
    frame.to_csv(OUT_CSV, index=False)


for seed in SEEDS:
    if (PROBAS / f"xlm-roberta-base_s{seed}_test.npy").exists():
        print(f"pomijam ziarno {seed} — policzone", flush=True)
        continue
    append_row(train_one(seed))

print(f"\nzapisano: {OUT_CSV}", flush=True)