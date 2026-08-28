"""combined condition: reweighting + augmentation.

Run from experiments/:  THESIS_HF_OUT=/var/tmp/thesis_hf ../.venv/bin/python 20_aug_combined.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.multiclass import OneVsRestClassifier

from thesis_lib import EMOTIONS, optimal_thresholds, bootstrap_f1_ci

RANDOM_STATE = 42
RARE = ["strach", "zaufanie", "smutek"]
FREQUENT = [e for e in EMOTIONS if e not in RARE]

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results"
# Checkpoints are throwaway; keep them off the NTFS symlink (torch.save corruption).
HF_OUT = Path(os.environ.get("THESIS_HF_OUT", "/var/tmp/thesis_hf"))
HF_OUT.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULTS / "aug_combined.csv"

tw_train = pd.read_csv(PROCESSED / "twitteremo_train.csv").reset_index(drop=True)
tw_val = pd.read_csv(PROCESSED / "twitteremo_val.csv").reset_index(drop=True)
tw_test = pd.read_csv(PROCESSED / "twitteremo_test.csv").reset_index(drop=True)
for df in (tw_train, tw_val, tw_test):
    df["tekst"] = df["tekst"].fillna("")
y_val, y_test = tw_val[EMOTIONS].values, tw_test[EMOTIONS].values

llm = pd.read_csv(PROCESSED / "aug_llm_train.csv")
llm["tekst"] = llm["tekst"].fillna("")
bt = pd.read_csv(PROCESSED / "aug_bt_train.csv")
bt["tekst"] = bt["tekst"].fillna("")

aug_llm = pd.concat([tw_train, llm[["tekst"] + EMOTIONS]], ignore_index=True)
aug_bt = pd.concat([tw_train, bt[["tekst"] + EMOTIONS]], ignore_index=True)


def rare_freq_row(pred: np.ndarray, row: dict) -> dict:
    """Attach per-rare-class recall/F1 and the frequent-class control average."""
    for c in RARE:
        i = EMOTIONS.index(c)
        row[f"recall_{c}"] = recall_score(y_test[:, i], pred[:, i], zero_division=0)
        row[f"f1_{c}"] = f1_score(y_test[:, i], pred[:, i], zero_division=0)
    row["f1_frequent_avg"] = float(np.mean(
        [f1_score(y_test[:, EMOTIONS.index(e)], pred[:, EMOTIONS.index(e)], zero_division=0)
         for e in FREQUENT]))
    return row


def classical_condition(name: str, train_df: pd.DataFrame) -> dict:
    """LogReg + char TF-IDF with class_weight='balanced' on the given train (recipe from 09)."""
    t0 = time.time()
    vec = TfidfVectorizer(max_features=50_000, ngram_range=(3, 5), analyzer="char_wb",
                          min_df=3, sublinear_tf=True, lowercase=True)
    X_tr = vec.fit_transform(train_df["tekst"].fillna(""))
    X_va, X_te = vec.transform(tw_val["tekst"]), vec.transform(tw_test["tekst"])
    clf = OneVsRestClassifier(LogisticRegression(
        max_iter=1000, C=1.0, class_weight="balanced",
        solver="liblinear", random_state=RANDOM_STATE))
    clf.fit(X_tr, train_df[EMOTIONS].values)
    thr = optimal_thresholds(y_val, clf.predict_proba(X_va))
    pred = (clf.predict_proba(X_te) >= thr).astype(int)
    base, lo, hi = bootstrap_f1_ci(y_test, pred)
    row = {"arm": "classical", "warunek": name, "n_train": len(train_df),
           "f1_macro": base, "ci_low": round(lo, 3), "ci_high": round(hi, 3),
           "f1_micro": f1_score(y_test, pred, average="micro", zero_division=0),
           "minutes": round((time.time() - t0) / 60, 1)}
    return rare_freq_row(pred, row)


def herbert_condition(name: str, train_df: pd.DataFrame, epochs: int = 4) -> dict:
    """HerBERT-base fine-tune with pos_weight from the given train (recipe from 10)."""
    import torch
    import torch.nn.functional as F
    from scipy.special import expit
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer, DataCollatorWithPadding,
                              EarlyStoppingCallback, set_seed)

    set_seed(RANDOM_STATE)
    t0 = time.time()
    model_name = "allegro/herbert-base-cased"
    tok = AutoTokenizer.from_pretrained(model_name)

    def to_ds(df: pd.DataFrame) -> Dataset:
        d = Dataset.from_dict({"text": df["tekst"].tolist(),
                               "labels": df[EMOTIONS].values.astype("float32").tolist()})
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=128),
                     batched=True, remove_columns=["text"])

    ds_train, ds_val, ds_test = to_ds(train_df), to_ds(tw_val), to_ds(tw_test)

    # pos_weight from the ACTUAL (augmented) train — inverse frequency of what the model sees.
    pos = train_df[EMOTIONS].values.sum(axis=0)
    neg = len(train_df) - pos
    pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 10.0), dtype=torch.float32)

    class WeightedTrainer(Trainer):
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

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(EMOTIONS), problem_type="multi_label_classification")
    args = TrainingArguments(
        output_dir=str(HF_OUT / f"aug20_{name}"),
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
        per_device_train_batch_size=8, per_device_eval_batch_size=32,
        gradient_accumulation_steps=2, gradient_checkpointing=True,
        num_train_epochs=epochs, learning_rate=2e-5, warmup_ratio=0.1, weight_decay=0.01,
        fp16=torch.cuda.is_available(), logging_steps=100, report_to="none", seed=RANDOM_STATE)

    def compute_metrics(p):
        pred = (expit(p.predictions) >= 0.5).astype(int)
        return {"f1_macro": f1_score(p.label_ids.astype(int), pred, average="macro", zero_division=0)}

    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
        pos_weight=pw, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train()

    p_val = expit(trainer.predict(ds_val).predictions)
    p_test = expit(trainer.predict(ds_test).predictions)
    thr = optimal_thresholds(y_val, p_val)
    pred = (p_test >= thr).astype(int)
    base, lo, hi = bootstrap_f1_ci(y_test, pred)
    row = {"arm": "herbert", "warunek": name, "n_train": len(train_df),
           "f1_macro": base, "ci_low": round(lo, 3), "ci_high": round(hi, 3),
           "f1_micro": f1_score(y_test, pred, average="micro", zero_division=0),
           "minutes": round((time.time() - t0) / 60, 1)}
    row = rare_freq_row(pred, row)

    del trainer, model
    torch.cuda.empty_cache()
    return row


def already_done() -> set[tuple[str, str]]:
    """Read existing CSV so re-launches skip completed (arm, condition) pairs."""
    if not OUT_CSV.exists():
        return set()
    df = pd.read_csv(OUT_CSV)
    return set(zip(df["arm"], df["warunek"]))


def append(row: dict) -> None:
    pd.DataFrame([row]).to_csv(OUT_CSV, mode="a", header=not OUT_CSV.exists(), index=False)
    print(f"  -> {row['arm']}/{row['warunek']}: F1-Macro={row['f1_macro']:.3f} "
          f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]  ({row['minutes']} min)", flush=True)


def main() -> None:
    done = already_done()
    jobs = [
        ("classical", "reweight+llm", lambda: classical_condition("reweight+llm", aug_llm)),
        ("classical", "reweight+bt", lambda: classical_condition("reweight+bt", aug_bt)),
        ("herbert", "reweight+llm", lambda: herbert_condition("reweight+llm", aug_llm)),
    ]
    for arm, name, fn in jobs:
        if (arm, name) in done:
            print(f"skip {arm}/{name} (already in CSV)", flush=True)
            continue
        print(f"=== {arm} / {name} ===", flush=True)
        append(fn())

    print("\n=== aug_combined.csv ===", flush=True)
    print(pd.read_csv(OUT_CSV).round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
