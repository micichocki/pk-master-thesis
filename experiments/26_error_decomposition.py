"""error decomposition of the classical model vs both HerBERT sizes.

Run from experiments/:  ../.venv/bin/python 26_error_decomposition.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier

from thesis_lib import EMOTIONS, RANDOM_STATE, optimal_thresholds

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results"
FEATURES = ROOT / "data" / "features"
FIGURES = ROOT / "thesis" / "images"

ENCODERS = {
    "HerBERT-base": RESULTS / "local_probas" / "herbert-base-cased",
    "HerBERT-large": RESULTS / "external_probas" / "herbert_large_full",
}
FIGURE_MODEL = "HerBERT-large"


def classical_predictions(y_train: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """Binary test predictions of the canonical classical model.

    Rebuilds the canonical recipe: character TF-IDF (cached in
    data/features), one-vs-rest logistic regression with balanced class weights
    and per-label thresholds tuned on the validation split.

    Args:
        y_train: Training label matrix.
        y_val: Validation label matrix, used for threshold tuning.

    Returns:
        Binary prediction matrix for the test split.
    """
    features = pickle.load(open(FEATURES / "TW_FEATURES.pkl", "rb"))["tfidf_char"]
    model = OneVsRestClassifier(LogisticRegression(
        max_iter=1000, C=1.0, class_weight="balanced",
        solver="liblinear", random_state=RANDOM_STATE))
    model.fit(features["train"], y_train)
    thresholds = optimal_thresholds(y_val, model.predict_proba(features["val"]))
    return (model.predict_proba(features["test"]) >= thresholds).astype(int)


def decompose(y_true: np.ndarray, pred_classical: np.ndarray,
              pred_encoder: np.ndarray) -> dict[str, float]:
    """Split every per-label decision into the four agreement cells.

    Args:
        y_true: Gold label matrix of the test split.
        pred_classical: Binary predictions of the classical model.
        pred_encoder: Binary predictions of the encoder.

    Returns:
        Counts and shares of repairs, classical-only wins and joint failures.
    """
    ok_c, ok_e = pred_classical == y_true, pred_encoder == y_true
    total, wrong_c = y_true.size, int((~ok_c).sum())
    repairs = int(((~ok_c) & ok_e).sum())
    classical_only = int((ok_c & (~ok_e)).sum())
    both_wrong = int(((~ok_c) & (~ok_e)).sum())
    disputed = repairs + classical_only
    return {
        "decyzje": total,
        "bledy_klasycznego": wrong_c,
        "naprawia": repairs,
        "naprawia_pct_bledow": round(100 * repairs / wrong_c, 1),
        "naprawia_pct_decyzji": round(100 * repairs / total, 1),
        "oba_zle": both_wrong,
        "oba_zle_pct_bledow": round(100 * both_wrong / wrong_c, 1),
        "oba_zle_pct": round(100 * both_wrong / total, 1),
        "klasyczny_lepszy": classical_only,
        "klasyczny_lepszy_pct": round(100 * classical_only / total, 1),
        "sporne": disputed,
        "sporne_enkoder_pct": round(100 * repairs / disputed, 1),
        "sporne_klasyczny_pct": round(100 * classical_only / disputed, 1),
        "bilans": repairs - classical_only,
        "netto_pp": round(100 * (repairs - classical_only) / total, 1),
    }


def per_emotion_figure(y_true: np.ndarray, pred_classical: np.ndarray,
                       pred_encoder: np.ndarray, encoder_name: str) -> None:
    """Redraw fig. 5.x: per-emotion F1 of both models, ordered by class support."""
    support = y_true.sum(axis=0)
    frame = pd.DataFrame({
        "emocja": EMOTIONS,
        "support": support,
        "Klasyczny": [f1_score(y_true[:, i], pred_classical[:, i], zero_division=0)
                      for i in range(len(EMOTIONS))],
        encoder_name: [f1_score(y_true[:, i], pred_encoder[:, i], zero_division=0)
                       for i in range(len(EMOTIONS))],
    }).sort_values("support")

    sns.set_theme(style="whitegrid")
    plt.rcParams["savefig.dpi"] = 300
    fig, ax = plt.subplots(figsize=(9, 5))
    x, width = np.arange(len(EMOTIONS)), 0.4
    ax.bar(x - width / 2, frame["Klasyczny"], width, label="Klasyczny", color="cornflowerblue")
    ax.bar(x + width / 2, frame[encoder_name], width, label=encoder_name, color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["emocja"] + " (n=" + frame["support"].astype(str) + ")",
                       rotation=30, ha="right")
    ax.set_ylabel("F1")
    ax.set_title(f"F1 per emocja (sortowane wg liczności) — klasyczny vs {encoder_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "err_per_emotion.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """Compute both decompositions, write artifacts and redraw the figure."""
    splits = {s: pd.read_csv(PROCESSED / f"twitteremo_{s}.csv") for s in ("train", "val", "test")}
    y_train, y_val, y_test = (splits[s][EMOTIONS].values for s in ("train", "val", "test"))

    pred_c = classical_predictions(y_train, y_val)
    f1_classical = f1_score(y_test, pred_c, average="macro", zero_division=0)
    print(f"klasyczny: F1-Macro={f1_classical:.3f}, błędnych decyzji={(pred_c != y_test).sum()}")

    rows, predictions = [], {}
    for name, prefix in ENCODERS.items():
        proba_val = np.load(f"{prefix}_proba_val.npy")
        proba_test = np.load(f"{prefix}_proba_test.npy")
        pred_e = (proba_test >= optimal_thresholds(y_val, proba_val)).astype(int)
        predictions[name] = pred_e
        per_emotion = [f1_score(y_test[:, i], pred_e[:, i], zero_division=0)
                       for i in range(len(EMOTIONS))]
        rows.append({
            "encoder": name,
            "f1_macro": round(f1_score(y_test, pred_e, average="macro", zero_division=0), 3),
            **decompose(y_test, pred_c, pred_e),
            "subset_acc_klasyczny": round((pred_c == y_test).all(axis=1).mean(), 3),
            "subset_acc_encoder": round((pred_e == y_test).all(axis=1).mean(), 3),
            "r_support_f1": round(float(np.corrcoef(y_test.sum(axis=0), per_emotion)[0, 1]), 3),
            "f1_strach": round(per_emotion[EMOTIONS.index("strach")], 3),
        })
        print(f"{name}: {rows[-1]}")

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "error_decomposition.csv", index=False)

    # Per-emotion F1 and the gain over the classical model, ordered by support:
    # F1-Macro is the mean of these columns, so the gains show which emotions
    # the encoder's advantage on the main metric actually comes from.
    per_label = pd.DataFrame({"emocja": EMOTIONS, "support": y_test.sum(axis=0)})
    per_label["f1_klasyczny"] = [round(f1_score(y_test[:, i], pred_c[:, i], zero_division=0), 3)
                                 for i in range(len(EMOTIONS))]
    for name, pred_e in predictions.items():
        per_label[f"f1_{name}"] = [round(f1_score(y_test[:, i], pred_e[:, i], zero_division=0), 3)
                                   for i in range(len(EMOTIONS))]
        per_label[f"zysk_{name}"] = (per_label[f"f1_{name}"] - per_label["f1_klasyczny"]).round(3)
    per_label.sort_values("support").to_csv(RESULTS / "error_per_emotion.csv", index=False)

    per_emotion_figure(y_test, pred_c, predictions[FIGURE_MODEL], FIGURE_MODEL)
    print(f"\nZapisano artefakty + rysunek err_per_emotion.pdf ({FIGURE_MODEL})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())