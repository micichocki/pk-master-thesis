"""Paired test for the stacking result.

Writes ``data/results/stacking_paired.csv`` and the probability matrices under
``data/results/classical_probas/``.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_lib import bootstrap_f1_ci, load_splits, optimal_thresholds, paired_bootstrap

EMOTIONS = ["radość", "smutek", "zaufanie", "wstręt", "strach", "gniew", "przeczuwanie", "zdziwienie"]
RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
PROBAS = ROOT / "data/results/classical_probas"
PROBAS.mkdir(parents=True, exist_ok=True)
OUT_CSV = ROOT / "data/results/stacking_paired.csv"

splits = load_splits("twitteremo", processed_dir=ROOT / "data/processed")
y_train, y_val, y_test = (splits[s].attrs["y"] for s in ("train", "val", "test"))

with open(ROOT / "data/features/TW_FEATURES.pkl", "rb") as fh:
    FEATURES = pickle.load(fh)


def logreg() -> LogisticRegression:
    """The canonical classical estimator (03b/03c)."""
    return LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                              solver="liblinear", random_state=RANDOM_STATE)


def fit_predict(estimator, feats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Fit one-vs-rest on train, return (val, test) probability matrices."""
    clf = OneVsRestClassifier(estimator, n_jobs=1)
    clf.fit(feats["train"], y_train)
    return clf.predict_proba(feats["val"]), clf.predict_proba(feats["test"])


def stacking_oof_multi(estimators: list[tuple[str, object, dict]], n_splits: int = 5
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold stacking features; each base learner uses its own representation.

    Mirrors ``stacking_oof_multi`` from notebook 03c, including the KFold seed, so the
    published figure (0.454 on test) reproduces bit for bit.
    """
    n_labels, n_est, n_train = y_train.shape[1], len(estimators), y_train.shape[0]
    oof_train = np.zeros((n_train, n_est * n_labels))
    oof_val = np.zeros((estimators[0][2]["val"].shape[0], n_est * n_labels))
    oof_test = np.zeros((estimators[0][2]["test"].shape[0], n_est * n_labels))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for ei, (name, est, feats) in enumerate(estimators):
        lo, hi = ei * n_labels, (ei + 1) * n_labels
        print(f"  baza: {name}", flush=True)
        X_tr, X_va, X_te = feats["train"], feats["val"], feats["test"]
        for tr_idx, va_idx in kf.split(np.arange(n_train)):
            clf = OneVsRestClassifier(est, n_jobs=1)
            clf.fit(X_tr[tr_idx], y_train[tr_idx])
            oof_train[va_idx, lo:hi] = clf.predict_proba(X_tr[va_idx])
        clf_full = OneVsRestClassifier(est, n_jobs=1)
        clf_full.fit(X_tr, y_train)
        oof_val[:, lo:hi] = clf_full.predict_proba(X_va)
        oof_test[:, lo:hi] = clf_full.predict_proba(X_te)
    return oof_train, oof_val, oof_test


print("=== model kanoniczny: LogReg + TF-IDF znakowy ===", flush=True)
p_val_single, p_test_single = fit_predict(logreg(), FEATURES["tfidf_char"])
thr_single = optimal_thresholds(y_val, p_val_single)
pred_single = (p_test_single >= thr_single).astype(int)

print("=== stacking v2: trzy silne bazy + meta-LogReg ===", flush=True)
base_v2 = [
    ("logreg_char", logreg(), FEATURES["tfidf_char"]),
    ("logreg_word", logreg(), FEATURES["tfidf_word"]),
    ("lightgbm_fasttext", lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, n_jobs=-1,
                                             class_weight="balanced",
                                             random_state=RANDOM_STATE, verbose=-1),
     FEATURES["fasttext"]),
]
oof_tr, oof_va, oof_te = stacking_oof_multi(base_v2)
meta = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE))
meta.fit(oof_tr, y_train)
p_val_stack, p_test_stack = meta.predict_proba(oof_va), meta.predict_proba(oof_te)
thr_stack = optimal_thresholds(y_val, p_val_stack)
pred_stack = (p_test_stack >= thr_stack).astype(int)

np.save(PROBAS / "single_proba_test.npy", p_test_single)
np.save(PROBAS / "single_proba_val.npy", p_val_single)
np.save(PROBAS / "stacking_proba_test.npy", p_test_stack)
np.save(PROBAS / "stacking_proba_val.npy", p_val_stack)

rows = []
for name, pred in (("model kanoniczny", pred_single), ("stacking v2", pred_stack)):
    base, lo, hi = bootstrap_f1_ci(y_test, pred)
    rows.append({"model": name, "f1_macro": round(base, 4),
                 "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
    print(f"{name:18s} F1-Macro {base:.4f} [{lo:.3f}; {hi:.3f}]", flush=True)

diff, lo, hi, p_better = paired_bootstrap(y_test, pred_single, pred_stack)
print(f"\nróżnica sparowana (stacking − kanoniczny): {diff:+.4f} [{lo:+.4f}; {hi:+.4f}], "
      f"P(stacking lepszy) = {p_better:.3f}")
istotna = not (lo < 0 < hi)
print("werdykt:", "różnica istotna" if istotna else "w granicach niepewności")

rows.append({"model": "różnica sparowana (stacking − kanoniczny)", "f1_macro": round(diff, 4),
             "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f"\nzapisano: {OUT_CSV}")