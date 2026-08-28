"""Measure the noise floor of a single fine-tune and decompose it per emotion.

Writes ``data/results/noise_floor.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_lib import load_splits, optimal_thresholds

EMOTIONS = ["radość", "smutek", "zaufanie", "wstręt", "strach", "gniew", "przeczuwanie", "zdziwienie"]
ROOT = Path(__file__).resolve().parent.parent
PROBAS = ROOT / "data/results/pos_weight_probas"
OUT_CSV = ROOT / "data/results/noise_floor.csv"
POOLED_CSV = ROOT / "data/results/noise_floor_pooled.csv"

splits = load_splits("twitteremo", processed_dir=ROOT / "data/processed")
y_val, y_test = splits["val"].attrs["y"], splits["test"].attrs["y"]

runs = sorted(p.stem[: -len("_test")] for p in PROBAS.glob("*_test.npy"))
if not runs:
    raise SystemExit(f"brak macierzy prawdopodobieństw w {PROBAS} — uruchom 32_pos_weight_verify.py")

macro: dict[str, float] = {}
per_class: dict[str, list[float]] = {}
for run in runs:
    thr = optimal_thresholds(y_val, np.load(PROBAS / f"{run}_val.npy"))
    pred = (np.load(PROBAS / f"{run}_test.npy") >= thr).astype(int)
    macro[run] = f1_score(y_test, pred, average="macro", zero_division=0)
    per_class[run] = [f1_score(y_test[:, i], pred[:, i], zero_division=0) for i in range(len(EMOTIONS))]

vals = np.array([macro[r] for r in runs])
same_cfg = np.array([macro[r] for r in runs if r.startswith("clip10")])  # identical configuration

print("F1-Macro, sześć niezależnych dostrajań:", np.round(vals, 4))
print(f"  odch. std {vals.std(ddof=1):.4f}, rozstęp {np.ptp(vals):.4f}")
print("F1-Macro, trzy runy TEJ SAMEJ konfiguracji (clip10):", np.round(same_cfg, 4))
print(f"  odch. std {same_cfg.std(ddof=1):.4f}, rozstęp {np.ptp(same_cfg):.4f}")

matrix = np.array([per_class[r] for r in runs])  # runs x emotions
var_class = matrix.var(axis=0, ddof=1)
var_macro = vals.var(ddof=1)
share = (var_class / len(EMOTIONS) ** 2) / var_macro

print(f"\nsuma Var(F1_i)/64 = {var_class.sum() / 64:.6f} wobec Var(F1-Macro) = {var_macro:.6f}"
      f" — człon kowariancyjny wynosi {var_macro - var_class.sum() / 64:+.6f}")

frame = pd.DataFrame({
    "emocja": EMOTIONS,
    "n_test": y_test.sum(axis=0).astype(int),
    "f1_min": matrix.min(axis=0).round(3),
    "f1_max": matrix.max(axis=0).round(3),
    "rozstep": (matrix.max(axis=0) - matrix.min(axis=0)).round(3),
    "std": matrix.std(axis=0, ddof=1).round(4),
    "udzial_wariancji": share.round(4),
}).sort_values("udzial_wariancji", ascending=False)
print("\n", frame.to_string(index=False), sep="")

frame.to_csv(OUT_CSV, index=False)
print(f"\nzapisano: {OUT_CSV}")

# --------------------------------------------------------------------------- #
# Pooled sigma across every replicated configuration (the figure quoted in ch. 3)
#
# A per-configuration sigma from n=3 is useless as a threshold: its own 95% CI spans
# more than an order of magnitude, and four of the six configurations below are the
# *same* model with estimates differing sixfold. Pooling the within-configuration
# deviations raises the degrees of freedom from 2 to 15 and tightens that CI to a
# factor of two, which is what makes the number usable as a decision rule.
# --------------------------------------------------------------------------- #
REPLICATED: dict[str, list[float]] = {
    "HerBERT-base, ważenie (clip10)": [0.5524, 0.5655, 0.5441],
    "HerBERT-base, ważenie bez przycięcia": [0.5532, 0.5469, 0.5581],
    "HerBERT-base, bez ważenia": [0.5404, 0.5194, 0.5651],
    "HerBERT-base, multiseed (kanon)": [0.5513, 0.5492, 0.5441],
    "XLM-R-base, multiseed": [0.5390, 0.5270, 0.5235],
    "HerBERT-large, czyste TW": [0.5770, 0.5823, 0.5900, 0.5920, 0.5950, 0.6050],
}

print("\n=== rozrzut per konfiguracja ===")
ss, dof, rows = 0.0, 0, []
for name, values in REPLICATED.items():
    arr = np.array(values)
    ss += ((arr - arr.mean()) ** 2).sum()
    dof += len(arr) - 1
    rows.append({"konfiguracja": name, "n": len(arr), "srednia": round(arr.mean(), 4),
                 "sigma": round(arr.std(ddof=1), 4), "rozstep": round(float(np.ptp(arr)), 4)})
    print(f"  {name:38s} n={len(arr)}  sigma={arr.std(ddof=1):.4f}  rozstęp={np.ptp(arr):.4f}")

pooled = float(np.sqrt(ss / dof))
lo_s = float(np.sqrt(dof * pooled**2 / chi2.ppf(0.975, dof)))
hi_s = float(np.sqrt(dof * pooled**2 / chi2.ppf(0.025, dof)))
band_single = 1.96 * pooled * np.sqrt(2)          # difference of two single runs
band_mean3 = 1.96 * pooled * np.sqrt(2 / 3)       # difference of two 3-seed means

print(f"\nsigma pooled = {pooled:.4f} ({dof} stopni swobody), 95% CI [{lo_s:.4f}; {hi_s:.4f}]")
print(f"pas 95% dla różnicy dwóch pojedynczych uruchomień: +/-{band_single:.4f}")
print(f"pas 95% dla różnicy dwóch średnich z 3 ziaren:      +/-{band_mean3:.4f}")

pd.DataFrame(rows + [{"konfiguracja": "POOLED", "n": dof + len(REPLICATED),
                      "srednia": None, "sigma": round(pooled, 4), "rozstep": None},
                     {"konfiguracja": "pas 95% (pojedyncze uruchomienia)", "n": None,
                      "srednia": None, "sigma": round(band_single, 4), "rozstep": None},
                     {"konfiguracja": "pas 95% (średnie z 3 ziaren)", "n": None,
                      "srednia": None, "sigma": round(band_mean3, 4), "rozstep": None},
                     ]).to_csv(POOLED_CSV, index=False)
print(f"zapisano: {POOLED_CSV}")
