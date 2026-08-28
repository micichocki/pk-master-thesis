"""Does removing the pos_weight cap really help, or was it seed noise?

Run from experiments/:
    WEIGHTING_HF_OUT=/var/tmp/pw_verify python 32_pos_weight_verify.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
from thesis_lib import (EMOTIONS, bootstrap_f1_ci, evaluate, make_weighted_trainer_cls,
                        optimal_thresholds, paired_bootstrap)

MODEL_NAME = "allegro/herbert-base-cased"
SEEDS = (42, 43, 44)
CONDITIONS = ("clip10", "bez_clip")
ROOT = Path(__file__).resolve().parent.parent
PROCESSED, RESULTS = ROOT / "data/processed", ROOT / "data/results"
HF_OUT = Path(os.environ.get("WEIGHTING_HF_OUT", "/var/tmp/pw_verify"))
PROBA_DIR = RESULTS / "pos_weight_probas"
HF_OUT.mkdir(parents=True, exist_ok=True)
PROBA_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULTS / "pos_weight_verify.csv"

splits = {s: pd.read_csv(PROCESSED / f"twitteremo_{s}.csv") for s in ("train", "val", "test")}
for frame in splits.values():
    frame["tekst"] = frame["tekst"].fillna("")
y = {s: splits[s][EMOTIONS].values for s in splits}

WeightedTrainer = make_weighted_trainer_cls()


def pos_weight_for(condition: str) -> torch.Tensor:
    """Per-label positive weight; ``clip10`` caps the ratio at 10, ``bez_clip`` does not."""
    pos = splits["train"][EMOTIONS].values.sum(axis=0)
    ratio = (len(splits["train"]) - pos) / np.maximum(pos, 1)
    ratio = np.clip(ratio, 1.0, 10.0) if condition == "clip10" else np.maximum(ratio, 1.0)
    return torch.tensor(ratio, dtype=torch.float32)


def run(condition: str, seed: int) -> None:
    """One fine-tune; saves metrics and the val/test probability matrices."""
    tag = f"{condition}_s{seed}"
    if OUT_CSV.exists() and tag in set(pd.read_csv(OUT_CSV)["run"]):
        print(f"== {tag}: już policzony, pomijam", flush=True)
        return

    set_seed(seed)
    pw = pos_weight_for(condition)
    print(f"\n=== {tag} (waga strachu {pw[EMOTIONS.index('strach')]:.1f}x) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def to_ds(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_dict({"text": df["tekst"].tolist(),
                                "labels": df[EMOTIONS].values.astype("float32").tolist()})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=128),
                      batched=True, remove_columns=["text"])

    ds_train, ds_val, ds_test = (to_ds(splits[s]) for s in ("train", "val", "test"))
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(EMOTIONS), problem_type="multi_label_classification")

    args = TrainingArguments(
        output_dir=str(HF_OUT / tag),
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
        per_device_train_batch_size=8, per_device_eval_batch_size=32,
        gradient_accumulation_steps=2, gradient_checkpointing=True,
        num_train_epochs=8, learning_rate=2e-5, warmup_ratio=0.1, weight_decay=0.01,
        fp16=torch.cuda.is_available(), logging_steps=200, report_to="none", seed=seed,
    )

    def compute_metrics(p) -> dict[str, float]:
        pred = (expit(p.predictions) >= 0.5).astype(int)
        return {"f1_macro": f1_score(p.label_ids.astype(int), pred, average="macro", zero_division=0)}

    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
        pos_weight=pw, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    started = time.time()
    trainer.train()

    p_val = expit(trainer.predict(ds_val).predictions)
    p_test = expit(trainer.predict(ds_test).predictions)
    np.save(PROBA_DIR / f"{tag}_val.npy", p_val)     # potrzebne do testu sparowanego
    np.save(PROBA_DIR / f"{tag}_test.npy", p_test)

    pred = (p_test >= optimal_thresholds(y["val"], p_val)).astype(int)
    metrics = evaluate(y["test"], pred)
    _, lo, hi = bootstrap_f1_ci(y["test"], pred)
    metrics.update({"run": tag, "warunek": condition, "seed": seed,
                    "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                    "minutes": round((time.time() - started) / 60, 1)})
    pd.DataFrame([metrics]).to_csv(OUT_CSV, mode="a", header=not OUT_CSV.exists(), index=False)
    print(f"  {tag}: F1-Macro={metrics['f1_macro']:.4f} [{lo:.3f}, {hi:.3f}]", flush=True)

    del trainer, model
    torch.cuda.empty_cache()


def predictions(tag: str) -> np.ndarray:
    """Thresholded test predictions of a finished run (thresholds from its own val)."""
    p_val = np.load(PROBA_DIR / f"{tag}_val.npy")
    p_test = np.load(PROBA_DIR / f"{tag}_test.npy")
    return (p_test >= optimal_thresholds(y["val"], p_val)).astype(int)


def report() -> None:
    """Paired bootstrap per seed and on seed-averaged probabilities."""
    res = pd.read_csv(OUT_CSV)
    print("\n=== wyniki pojedynczych runów ===")
    print(res[["run", "f1_macro", "ci_low", "ci_high", "minutes"]].round(4).to_string(index=False))

    print("\n=== test sparowany (bez_clip - clip10), per ziarno ===")
    for seed in SEEDS:
        tags = [f"{c}_s{seed}" for c in CONDITIONS]
        if not all((PROBA_DIR / f"{t}_test.npy").exists() for t in tags):
            print(f"  seed {seed}: brak kompletu runów")
            continue
        diff, lo, hi, _ = paired_bootstrap(y["test"], predictions(tags[0]), predictions(tags[1]))
        istotne = "ISTOTNE" if lo > 0 or hi < 0 else "w granicach szumu"
        print(f"  seed {seed}: {diff:+.4f} [{lo:+.4f}; {hi:+.4f}]  -> {istotne}")

    have = [s for s in SEEDS if all((PROBA_DIR / f"{c}_s{s}_test.npy").exists() for c in CONDITIONS)]
    if len(have) > 1:
        print("\n=== test sparowany na uśrednionych prawdopodobieństwach ===")
        preds = {}
        for c in CONDITIONS:
            pv = np.mean([np.load(PROBA_DIR / f"{c}_s{s}_val.npy") for s in have], axis=0)
            pt = np.mean([np.load(PROBA_DIR / f"{c}_s{s}_test.npy") for s in have], axis=0)
            preds[c] = (pt >= optimal_thresholds(y["val"], pv)).astype(int)
        diff, lo, hi, _ = paired_bootstrap(y["test"], preds["clip10"], preds["bez_clip"])
        istotne = "ISTOTNE" if lo > 0 or hi < 0 else "w granicach szumu"
        print(f"  {len(have)} ziaren: {diff:+.4f} [{lo:+.4f}; {hi:+.4f}]  -> {istotne}")
        print("\nWniosek wolno ogłosić TYLKO gdy przedział nie obejmuje zera "
              "i kierunek jest zgodny na wszystkich ziarnach.")


def main() -> None:
    for seed in SEEDS:
        for condition in CONDITIONS:
            run(condition, seed)
    report()


if __name__ == "__main__":
    main()