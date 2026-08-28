"""Paired bootstrap for every comparison reported as a tie.

Run from experiments/:  ../.venv/bin/python 25_paired_tests.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from thesis_lib import (EMOTIONS, bootstrap_f1_ci, optimal_thresholds,
                        paired_bootstrap)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results"
TRANSFORMERS = ROOT / "data" / "transformers"
LOCAL_PROBAS = RESULTS / "local_probas"
EXT_PROBAS = RESULTS / "external_probas"
LOCAL_PROBAS.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Local checkpoints (name -> base model for the tokenizer / LoRA adapter host).
LOCAL_MODELS: dict[str, str] = {
    "herbert-base-cased": "allegro/herbert-base-cased",
    "xlm-roberta-base": "FacebookAI/xlm-roberta-base",
    "herbert-large-cased-lora": "allegro/herbert-large-cased",
}
CLOUD_MODELS = ["herbert_large_full", "xlmr_large_full", "bielik45"]
ENCODERS = list(LOCAL_MODELS) + ["herbert_large_full", "xlmr_large_full"]

# Pairs reported as ties: (A, B, where the claim sits).
PAIRS: list[tuple[str, str, str]] = [
    ("xlm-roberta-base", "herbert-base-cased", "5.2.1 — natywny vs wielojęzyczny (base)"),
    ("herbert-large-cased-lora", "herbert_large_full", "5.2.2 — LoRA vs pełne dostrajanie"),
    ("xlmr_large_full", "herbert_large_full", "5.2.2 — natywny vs wielojęzyczny (large)"),
    ("herbert_large_full", "bielik45", "5.3.1 — Bielik vs najlepszy enkoder"),
    ("bielik45", "ensemble6", "5.5 — zespół vs najlepszy składnik"),
    ("herbert_large_full", "ensemble6", "5.6 — zespół vs najlepszy enkoder"),
    ("ensemble5", "ensemble6", "5.5 — wpływ dołączenia Bielika"),
]


def latest_checkpoint(name: str) -> str:
    """Path of the highest-numbered checkpoint directory of a local run.

    Args:
        name: Sub-directory of data/transformers holding the run.

    Returns:
        Path to the checkpoint directory.

    Raises:
        FileNotFoundError: If the run has no checkpoint-* directory.
    """
    found = sorted(glob.glob(str(TRANSFORMERS / name / "checkpoint-*")))
    if not found:
        raise FileNotFoundError(f"brak checkpointu dla {name}")
    return found[-1]


@torch.no_grad()
def predict_proba(model, tokenizer, texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Per-label sigmoid probabilities for a list of texts.

    Args:
        model: A sequence-classification model in eval mode on ``DEVICE``.
        tokenizer: Matching tokenizer.
        texts: Raw texts to score.
        batch_size: Texts per forward pass.

    Returns:
        Array of shape ``(len(texts), len(EMOTIONS))``.
    """
    out: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start:start + batch_size], truncation=True, max_length=128,
                        padding=True, return_tensors="pt").to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=(DEVICE == "cuda")):
            out.append(expit(model(**enc).logits.float().cpu().numpy()))
    return np.vstack(out)


def local_probas(name: str, base: str, val_texts: list[str],
                 test_texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Val/test probabilities of a local checkpoint, cached on disk.

    Args:
        name: Run name under data/transformers.
        base: Base model id (tokenizer, and adapter host for LoRA runs).
        val_texts: Texts of the validation split.
        test_texts: Texts of the test split.

    Returns:
        Tuple ``(proba_val, proba_test)``.
    """
    cached_val = LOCAL_PROBAS / f"{name}_proba_val.npy"
    cached_test = LOCAL_PROBAS / f"{name}_proba_test.npy"
    if cached_val.exists() and cached_test.exists():
        print(f"  {name}: z cache")
        return np.load(cached_val), np.load(cached_test)

    checkpoint = latest_checkpoint(name)
    tokenizer = AutoTokenizer.from_pretrained(base)
    if "lora" in name:
        from peft import PeftModel
        model = AutoModelForSequenceClassification.from_pretrained(
            base, num_labels=len(EMOTIONS), problem_type="multi_label_classification")
        model = PeftModel.from_pretrained(model, checkpoint)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model = model.to(DEVICE).eval()

    proba_val = predict_proba(model, tokenizer, val_texts)
    proba_test = predict_proba(model, tokenizer, test_texts)
    np.save(cached_val, proba_val)
    np.save(cached_test, proba_test)
    del model
    torch.cuda.empty_cache()
    print(f"  {name}: policzone i zapisane")
    return proba_val, proba_test


def main() -> int:
    """Compute every paired test and write the result artifacts."""
    val = pd.read_csv(PROCESSED / "twitteremo_val.csv").reset_index(drop=True)
    test = pd.read_csv(PROCESSED / "twitteremo_test.csv").reset_index(drop=True)
    for df in (val, test):
        df["tekst"] = df["tekst"].fillna("")
    y_val, y_test = val[EMOTIONS].values, test[EMOTIONS].values

    proba_val: dict[str, np.ndarray] = {}
    proba_test: dict[str, np.ndarray] = {}

    print("Prawdopodobieństwa modeli lokalnych:")
    for name, base in LOCAL_MODELS.items():
        proba_val[name], proba_test[name] = local_probas(
            name, base, val["tekst"].tolist(), test["tekst"].tolist())

    print("Prawdopodobieństwa modeli chmurowych:")
    for name in CLOUD_MODELS:
        proba_val[name] = np.load(EXT_PROBAS / f"{name}_proba_val.npy")
        proba_test[name] = np.load(EXT_PROBAS / f"{name}_proba_test.npy")
        print(f"  {name}: wczytane")

    # Ensembles: unweighted mean of member probabilities (soft voting).
    proba_val["ensemble5"] = np.mean([proba_val[m] for m in ENCODERS], axis=0)
    proba_test["ensemble5"] = np.mean([proba_test[m] for m in ENCODERS], axis=0)
    proba_val["ensemble6"] = np.mean([proba_val[m] for m in ENCODERS + ["bielik45"]], axis=0)
    proba_test["ensemble6"] = np.mean([proba_test[m] for m in ENCODERS + ["bielik45"]], axis=0)

    # Thresholds tuned per system on val, then applied once to test.
    preds: dict[str, np.ndarray] = {}
    scores: dict[str, tuple[float, float, float]] = {}
    print("\nWyniki pojedynczych systemów (test):")
    for name in proba_val:
        thresholds = optimal_thresholds(y_val, proba_val[name])
        preds[name] = (proba_test[name] >= thresholds).astype(int)
        scores[name] = bootstrap_f1_ci(y_test, preds[name])
        f1, lo, hi = scores[name]
        print(f"  {name:26s} {f1:.3f} [{lo:.3f}; {hi:.3f}]")

    rows: list[dict[str, object]] = []
    print("\nTesty sparowane (Δ = B − A, 2000 resampli):")
    for name_a, name_b, claim in PAIRS:
        delta, lo, hi, p_better = paired_bootstrap(y_test, preds[name_a], preds[name_b])
        tie = lo < 0.0 < hi
        rows.append({
            "twierdzenie": claim,
            "model_A": name_a,
            "model_B": name_b,
            "f1_A": round(scores[name_a][0], 3),
            "f1_B": round(scores[name_b][0], 3),
            "delta": round(delta, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "p_B_lepszy": round(p_better, 3),
            "remis": tie,
        })
        verdict = "REMIS (CI zawiera 0)" if tie else "ISTOTNE"
        print(f"  {name_b} − {name_a}: Δ={delta:+.4f} [{lo:+.4f}; {hi:+.4f}] "
              f"P(B>A)={p_better:.2f}  {verdict}")

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "paired_tests.csv", index=False)

    print(f"\nZapisano: {RESULTS/'paired_tests.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())