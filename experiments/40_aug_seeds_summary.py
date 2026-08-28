"""Aggregate the HerBERT-large augmentation ablation over seeds 42/43/44.

Writes ``data/results/aug_seeds_summary.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_lib import load_splits, optimal_thresholds, paired_bootstrap

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data/results"
OUT_CSV = RESULTS / "aug_seeds_summary.csv"
KERNELS = {
    42: [ROOT / "data/results/aug_large_ablation.csv"],
    43: [ROOT / f"kaggle/herbert_large_aug_s43_{p}/output/result.csv" for p in "ab"],
    44: [ROOT / f"kaggle/herbert_large_aug_s44_{p}/output/result.csv" for p in "ab"],
}
PROBA_DIRS = {seed: [ROOT / f"kaggle/herbert_large_aug_s{seed}_{p}/output" for p in "ab"]
              for seed in (43, 44)}

splits = load_splits("twitteremo", processed_dir=ROOT / "data/processed")
y_val, y_test = splits["val"].attrs["y"], splits["test"].attrs["y"]

frames = []
for seed, paths in KERNELS.items():
    for path in paths:
        part = pd.read_csv(path)[["warunek", "f1_macro"]]
        part["seed"] = seed
        frames.append(part)
long = pd.concat(frames, ignore_index=True)

wide = long.pivot(index="warunek", columns="seed", values="f1_macro").sort_index()
wide["srednia"] = wide[[42, 43, 44]].mean(axis=1)
wide["odch"] = wide[[42, 43, 44]].std(axis=1, ddof=1)
wide["rozstep"] = wide[[42, 43, 44]].max(axis=1) - wide[[42, 43, 44]].min(axis=1)

# Pooled within-condition sigma: a per-condition sigma from three runs is far too
# uncertain to compare between rows, so the spread is estimated jointly.
resid = wide[[42, 43, 44]].sub(wide["srednia"], axis=0)
dof = len(wide) * 2
pooled = float(np.sqrt((resid**2).to_numpy().sum() / dof))
band = student_t.ppf(0.975, dof) * pooled * np.sqrt(2 / 3)

print(wide.round(4).to_string())
print(f"\nsigma pooled = {pooled:.4f} ({dof} stopni swobody)")
print(f"pas 95% dla różnicy dwóch średnich z 3 ziaren: +/-{band:.4f}")
print("\nranking wg średniej:")
for warunek, row in wide.sort_values("srednia", ascending=False).iterrows():
    delta = row["srednia"] - wide.loc["1_baseline", "srednia"]
    verdict = "poza pasem" if abs(delta) > band else "remis"
    print(f"  {warunek:20s} {row['srednia']:.4f}  wobec baseline {delta:+.4f}  {verdict}")


def averaged_predictions(warunek: str) -> np.ndarray | None:
    """Average val/test probabilities over seeds 43-44, then threshold on val."""
    vals, tests = [], []
    for seed, dirs in PROBA_DIRS.items():
        for d in dirs:
            v, t_ = d / f"proba_val_{warunek}.npy", d / f"proba_test_{warunek}.npy"
            if v.exists() and t_.exists():
                vals.append(np.load(v))
                tests.append(np.load(t_))
    if len(vals) < 2:
        return None
    thr = optimal_thresholds(y_val, np.mean(vals, axis=0))
    return (np.mean(tests, axis=0) >= thr).astype(int)

print("\ntest sparowany na uśrednionych prawdopodobieństwach (ziarna 43-44) wobec baseline:")
base_pred = averaged_predictions("1_baseline")
rows = []
for warunek in wide.index:
    row = {"warunek": warunek, "srednia": round(wide.loc[warunek, "srednia"], 4),
           "odch": round(wide.loc[warunek, "odch"], 4),
           "min": round(wide.loc[warunek, [42, 43, 44]].min(), 4),
           "max": round(wide.loc[warunek, [42, 43, 44]].max(), 4),
           "sigma_pooled": round(pooled, 4), "pas95": round(band, 4)}
    pred = averaged_predictions(warunek)
    if warunek != "1_baseline" and pred is not None and base_pred is not None:
        diff, lo, hi, p_better = paired_bootstrap(y_test, base_pred, pred)
        row.update({"roznica": round(diff, 4), "ci_low": round(lo, 4),
                    "ci_high": round(hi, 4), "p_lepszy": round(p_better, 3)})
        print(f"  {warunek:20s} {diff:+.4f} [{lo:+.4f}; {hi:+.4f}]  "
              f"P(lepszy) = {p_better:.3f}")
    rows.append(row)

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f"\nzapisano: {OUT_CSV}")
