"""Hyperparameter sensitivity of the classical final model.

Run from experiments/:  ../.venv/bin/python 23_hpo_classical.py
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from thesis_lib import (EMOTIONS, RANDOM_STATE, evaluate, optimal_thresholds,
                        bootstrap_f1_ci, paired_bootstrap)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]
NGRAM_GRID = [(2, 4), (3, 5), (3, 6)]
MAXFEAT_GRID = [30_000, 50_000, 100_000]
SUBLINEAR_GRID = [True, False]
CANONICAL = {"C": 1.0, "ngram": (3, 5), "max_features": 50_000, "sublinear_tf": True}

tw_train = pd.read_csv(PROCESSED / "twitteremo_train.csv").reset_index(drop=True)
tw_val = pd.read_csv(PROCESSED / "twitteremo_val.csv").reset_index(drop=True)
tw_test = pd.read_csv(PROCESSED / "twitteremo_test.csv").reset_index(drop=True)
for df in (tw_train, tw_val, tw_test):
    df["tekst"] = df["tekst"].fillna("")
y_train, y_val, y_test = (d[EMOTIONS].values for d in (tw_train, tw_val, tw_test))


def fit_features(ngram: tuple[int, int], max_features: int,
                 sublinear_tf: bool) -> dict[str, np.ndarray]:
    """Canonical char_wb vectorizer from 03a with the three tuned knobs exposed."""
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram,
                          analyzer="char_wb", min_df=3,
                          sublinear_tf=sublinear_tf, lowercase=True)
    return {"train": vec.fit_transform(tw_train["tekst"]),
            "val": vec.transform(tw_val["tekst"]),
            "test": vec.transform(tw_test["tekst"])}


def fit_config(X: dict, C: float) -> tuple[OneVsRestClassifier, np.ndarray]:
    """Fit OvR LogReg and tune per-label thresholds on val (canonical protocol)."""
    clf = OneVsRestClassifier(LogisticRegression(
        max_iter=1000, C=C, class_weight="balanced",
        solver="liblinear", random_state=RANDOM_STATE))
    clf.fit(X["train"], y_train)
    thr = optimal_thresholds(y_val, clf.predict_proba(X["val"]))
    return clf, thr


def main() -> None:
    rows = []
    vec_combos = list(itertools.product(NGRAM_GRID, MAXFEAT_GRID, SUBLINEAR_GRID))
    total = len(vec_combos) * len(C_GRID)
    done = 0
    t_start = time.time()
    for ngram, max_features, sublinear_tf in vec_combos:
        X = fit_features(ngram, max_features, sublinear_tf)
        for C in C_GRID:
            t0 = time.time()
            clf, thr = fit_config(X, C)
            pred_val = (clf.predict_proba(X["val"]) >= thr).astype(int)
            m = evaluate(y_val, pred_val)
            rows.append({"C": C, "ngram": f"({ngram[0]},{ngram[1]})",
                         "max_features": max_features, "sublinear_tf": sublinear_tf,
                         "n_features": X["train"].shape[1],
                         "f1_macro_val": round(m["f1_macro"], 4),
                         "f1_micro_val": round(m["f1_micro"], 4),
                         "time_s": round(time.time() - t0, 1)})
            done += 1
            print(f"[{done:2d}/{total}] C={C:<4} ngram=({ngram[0]},{ngram[1]}) "
                  f"maxf={max_features//1000}k sub={int(sublinear_tf)} "
                  f"-> F1-Macro(val)={m['f1_macro']:.4f}", flush=True)

    grid = pd.DataFrame(rows).sort_values("f1_macro_val", ascending=False).reset_index(drop=True)
    grid.to_csv(RESULTS / "hpo_classical.csv", index=False)
    print(f"\nGrid done in {(time.time() - t_start) / 60:.1f} min. Top 5 (val):")
    print(grid.head(5).to_string(index=False))

    # --- Final: canonical vs best-on-val, single test evaluation each ---------
    best = grid.iloc[0]
    best_ngram = tuple(int(x) for x in best["ngram"].strip("()").split(","))
    configs = {
        "kanoniczna (03a/03b)": CANONICAL,
        "strojona (best val)": {"C": float(best["C"]), "ngram": best_ngram,
                                "max_features": int(best["max_features"]),
                                "sublinear_tf": bool(best["sublinear_tf"])},
    }
    test_rows, preds = [], {}
    for name, cfg in configs.items():
        X = fit_features(cfg["ngram"], cfg["max_features"], cfg["sublinear_tf"])
        clf, thr = fit_config(X, cfg["C"])
        pred = (clf.predict_proba(X["test"]) >= thr).astype(int)
        preds[name] = pred
        base, lo, hi = bootstrap_f1_ci(y_test, pred)
        m = evaluate(y_test, pred)
        val_f1 = grid.loc[
            (grid["C"] == cfg["C"]) & (grid["ngram"] == f"({cfg['ngram'][0]},{cfg['ngram'][1]})")
            & (grid["max_features"] == cfg["max_features"])
            & (grid["sublinear_tf"] == cfg["sublinear_tf"]), "f1_macro_val"].iloc[0]
        test_rows.append({"konfiguracja": name, **{k: str(v) for k, v in cfg.items()},
                          "f1_macro_val": val_f1, "f1_macro_test": round(base, 3),
                          "ci95": f"[{lo:.3f}, {hi:.3f}]",
                          "f1_micro_test": round(m["f1_micro"], 3)})
        print(f"\n{name}: F1-Macro(test)={base:.3f} [{lo:.3f}, {hi:.3f}]")

    md, lo, hi, p_b = paired_bootstrap(y_test, preds["kanoniczna (03a/03b)"],
                                       preds["strojona (best val)"])
    sig = "TAK" if (lo > 0 or hi < 0) else "nie"
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(RESULTS / "hpo_classical_test.csv", index=False)
    print(f"\nPaired bootstrap (strojona - kanoniczna): delta={md:+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}], P(strojona>kanoniczna)={p_b:.3f}, istotne: {sig}")

if __name__ == "__main__":
    main()