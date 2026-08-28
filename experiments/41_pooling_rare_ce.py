"""Split the targeted pooling variant: rare rows from CLARIN-Emo alone.

Writes ``data/results/pooling_rare_ce.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_lib import bootstrap_f1_ci, optimal_thresholds

EMOTIONS = ["radość", "smutek", "zaufanie", "wstręt", "strach", "gniew", "przeczuwanie", "zdziwienie"]
RARE = ["strach", "zaufanie", "smutek"]
FREQUENT = [e for e in EMOTIONS if e not in RARE]
RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parent.parent
PROCESSED, RESULTS = ROOT / "data/processed", ROOT / "data/results"
OUT_CSV = RESULTS / "pooling_rare_ce.csv"


def load(prefix: str, text_col: str) -> dict[str, pd.DataFrame]:
    """Load the three splits of a corpus, normalising the text column name."""
    out = {}
    for split in ("train", "val", "test"):
        frame = pd.read_csv(PROCESSED / f"{prefix}_{split}.csv").reset_index(drop=True)
        frame["__text"] = frame[text_col].fillna("")
        out[split] = frame
    return out


TW, CE, GO = load("twitteremo", "tekst"), load("clarin_emo", "tekst"), load("go_emotions", "text_pl")
y_val, y_test = TW["val"][EMOTIONS].values, TW["test"][EMOTIONS].values


def train_eval(train_texts: list[str], train_y: np.ndarray) -> dict[str, float]:
    """Fit on the pooled train set; tune thresholds on TW val; evaluate on TW test."""
    vec = TfidfVectorizer(max_features=50_000, ngram_range=(3, 5), analyzer="char_wb",
                          min_df=3, sublinear_tf=True)
    x_train = vec.fit_transform(train_texts)
    x_val, x_test = vec.transform(TW["val"]["__text"]), vec.transform(TW["test"]["__text"])
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                                                 solver="liblinear", random_state=RANDOM_STATE))
    clf.fit(x_train, train_y)
    thr = optimal_thresholds(y_val, clf.predict_proba(x_val))
    pred = (clf.predict_proba(x_test) >= thr).astype(int)
    base, lo, hi = bootstrap_f1_ci(y_test, pred)
    row = {"n_train": len(train_texts), "f1_macro": round(base, 4),
           "ci_low": round(lo, 3), "ci_high": round(hi, 3)}
    row["f1_rare_avg"] = float(np.mean(
        [f1_score(y_test[:, EMOTIONS.index(e)], pred[:, EMOTIONS.index(e)], zero_division=0)
         for e in RARE]))
    row["f1_frequent_avg"] = float(np.mean(
        [f1_score(y_test[:, EMOTIONS.index(e)], pred[:, EMOTIONS.index(e)], zero_division=0)
         for e in FREQUENT]))
    return row


def pool(*frames: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Concatenate training frames into (texts, labels)."""
    texts = pd.concat([f["__text"] for f in frames], ignore_index=True).tolist()
    labels = np.vstack([f[EMOTIONS].values for f in frames])
    return texts, labels


rare_ce = CE["train"][CE["train"][RARE].sum(axis=1) > 0]
rare_go = GO["train"][GO["train"][RARE].sum(axis=1) > 0]
print(f"wierszy z klasą rzadką: CE {len(rare_ce)}, GO {len(rare_go)}", flush=True)

variants = {
    "1_TW": pool(TW["train"]),
    "2_TW+CE": pool(TW["train"], CE["train"]),
    "5_TW+rare(CE,GO)": pool(TW["train"], rare_ce, rare_go),
    "6_TW+rare(CE)": pool(TW["train"], rare_ce),
}

rows = []
for name, (texts, labels) in variants.items():
    row = train_eval(texts, labels)
    row["wariant"] = name
    rows.append(row)
    print(f"{name:20s} n={row['n_train']:6d}  F1-Macro={row['f1_macro']:.4f} "
          f"[{row['ci_low']:.3f}; {row['ci_high']:.3f}]  rzadkie={row['f1_rare_avg']:.3f}",
          flush=True)

frame = pd.DataFrame(rows)[["wariant", "n_train", "f1_macro", "ci_low", "ci_high",
                            "f1_rare_avg", "f1_frequent_avg"]]
frame.to_csv(OUT_CSV, index=False)
print(f"\nzapisano: {OUT_CSV}")
