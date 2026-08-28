"""Interpretability — Part A (classical: LogReg + char TF-IDF).

Run from repo root:  .venv/bin/python experiments/18_interpretability.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results"
FIGURES = ROOT / "thesis" / "images"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
EMOTIONS = ["radość", "smutek", "zaufanie", "wstręt", "strach", "gniew", "przeczuwanie", "zdziwienie"]

# ---------------------------------------------------------------- data + model
tw_train = pd.read_csv(PROCESSED / "twitteremo_train.csv")
tw_val = pd.read_csv(PROCESSED / "twitteremo_val.csv")
tw_test = pd.read_csv(PROCESSED / "twitteremo_test.csv")
for df in (tw_train, tw_val, tw_test):
    df["tekst"] = df["tekst"].fillna("")

y_train = tw_train[EMOTIONS].values
y_val = tw_val[EMOTIONS].values
y_test = tw_test[EMOTIONS].values

vec = TfidfVectorizer(
    max_features=50_000, ngram_range=(3, 5), analyzer="char_wb",
    min_df=3, sublinear_tf=True, lowercase=True,
)
X_train = vec.fit_transform(tw_train["tekst"])
X_val = vec.transform(tw_val["tekst"])
X_test = vec.transform(tw_test["tekst"])
feat_names = np.array(vec.get_feature_names_out())
print(f"[features] train={X_train.shape}  vocab={len(feat_names)}")

clf = OneVsRestClassifier(LogisticRegression(
    max_iter=1000, C=1.0, class_weight="balanced",
    solver="liblinear", random_state=RANDOM_STATE,
))
clf.fit(X_train, y_train)

# thresholds: canonical values from exp3 (per-emotion, tuned on val)
thr_df = pd.read_csv(RESULTS / "exp3_thresholds.csv").set_index("emotion")
thresholds = np.array([thr_df.loc[e, "optimal_threshold"] for e in EMOTIONS])

proba_test = clf.predict_proba(X_test)
pred_test = (proba_test >= thresholds).astype(int)
f1m = f1_score(y_test, pred_test, average="macro", zero_division=0)
print(f"[check] classical test F1-Macro = {f1m:.4f}  (expected ~0.474)")

# coefficient matrix: (n_emotions, vocab)
coef = np.vstack([est.coef_.ravel() for est in clf.estimators_])

# persist vectorizer + coef so the paired renderer can reuse the classical model
import pickle
with open(ROOT / "data" / "features" / "interp_classical_model.pkl", "wb") as f:
    pickle.dump({"vectorizer": vec, "coef": coef, "emotions": EMOTIONS}, f)

# ---------------------------------------------------------------- global importance
rows = []
TOPK = 15
# Exact linear SHAP (closed form): phi_ij = coef_j * (x_ij - mu_j), mu = train mean.
# mean_i |phi_ij| = |coef_j| * mean_i |x_ij - mu_j|. Sparse-aware identity below avoids
# densifying X_test: docs where n-gram j is absent contribute |0 - mu_j| = mu_j each.
mu = np.asarray(X_train.mean(axis=0)).ravel()
Xc = X_test.tocsc()
n_docs = X_test.shape[0]
nnz_per_col = np.diff(Xc.indptr)
col_of_nnz = np.repeat(np.arange(Xc.shape[1]), nnz_per_col)
sum_abs_dev = np.bincount(col_of_nnz, weights=np.abs(Xc.data - mu[col_of_nnz]),
                          minlength=Xc.shape[1])
mean_abs_dev = (sum_abs_dev + (n_docs - nnz_per_col) * mu) / n_docs
mean_abs_shap = np.abs(coef) * mean_abs_dev          # (n_emotions, vocab)

for e_idx, emo in enumerate(EMOTIONS):
    top_coef = np.argsort(coef[e_idx])[::-1][:TOPK]
    top_shap = np.argsort(mean_abs_shap[e_idx])[::-1][:TOPK]
    for rank, fi in enumerate(top_coef):
        rows.append({"emotion": emo, "kind": "coef", "rank": rank + 1,
                     "ngram": feat_names[fi], "value": float(coef[e_idx, fi])})
    for rank, fi in enumerate(top_shap):
        rows.append({"emotion": emo, "kind": "mean_abs_shap", "rank": rank + 1,
                     "ngram": feat_names[fi], "value": float(mean_abs_shap[e_idx, fi])})
global_df = pd.DataFrame(rows)
global_df.to_csv(RESULTS / "interp_classical_global_top_ngrams.csv", index=False)
print(f"[save] {RESULTS / 'interp_classical_global_top_ngrams.csv'}")

f2i = {f: i for i, f in enumerate(feat_names)}


def global_figure(kind: str, value_label: str, suptitle: str, path: Path,
                  signed_colors: bool) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    for ax, (e_idx, emo) in zip(axes.ravel(), enumerate(EMOTIONS)):
        sub = global_df[(global_df.emotion == emo) & (global_df.kind == kind)].head(12).iloc[::-1]
        labels = [repr(s)[1:-1] for s in sub.ngram]  # show spaces visibly
        if signed_colors:
            colors = ["#c0392b" if coef[e_idx, f2i[ng]] > 0 else "#2980b9" for ng in sub.ngram]
        else:
            colors = "#c0392b"
        ax.barh(range(len(sub)), sub.value, color=colors)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(labels, fontsize=8, family="monospace")
        ax.set_title(emo, fontsize=11)
        ax.set_xlabel(value_label, fontsize=8)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)
    print(f"[save] {path}")


global_figure("coef", "waga LogReg",
              "Najważniejsze char n-gramy per emocja (LogReg + TF-IDF char)",
              FIGURES / "interp_classical_global.pdf", signed_colors=False)
global_figure("mean_abs_shap", "średni |SHAP| (test)",
              "Najważniejsze char n-gramy per emocja — SHAP (postać zamknięta dla modelu liniowego)\n"
              "kolor: czerwony = za emocją, niebieski = przeciw; ranking zdominowany przez częste n-gramy",
              FIGURES / "interp_classical_global_shap.pdf", signed_colors=True)

# ---------------------------------------------------------------- select sentences
# Deterministic, illustrative cases shared with the HerBERT kernel.
def pick(mask: np.ndarray, score: np.ndarray, n: int = 1) -> list[int]:
    idx = np.where(mask)[0]
    return idx[np.argsort(score[idx])[::-1][:n]].tolist()

n_emo = y_test.sum(axis=1)
conf = proba_test.max(axis=1)
selected: dict[str, int] = {}
# one confident single-emotion example for 4 emotions (incl. rare strach)
for emo in ["radość", "wstręt", "strach", "przeczuwanie"]:
    e_idx = EMOTIONS.index(emo)
    mask = (y_test[:, e_idx] == 1) & (n_emo == 1)
    got = pick(mask, proba_test[:, e_idx], 1)
    if got:
        selected[f"single_{emo}"] = got[0]
# one sarcastic example (column sarkazm) that is positive on >=1 emotion
sark = tw_test["sarkazm"].values if "sarkazm" in tw_test else np.zeros(len(tw_test))
got = pick((sark == 1) & (n_emo >= 1), conf, 1)
if got:
    selected["sarkazm"] = got[0]
# one multi-label example (>=3 emotions)
got = pick(n_emo >= 3, conf, 1)
if got:
    selected["multi"] = got[0]

sel_records = []
for tag, i in selected.items():
    sel_records.append({
        "tag": tag, "test_index": int(i), "tekst": tw_test["tekst"].iloc[i],
        "true": [EMOTIONS[k] for k in range(len(EMOTIONS)) if y_test[i, k] == 1],
        "classical_pred": [EMOTIONS[k] for k in range(len(EMOTIONS)) if pred_test[i, k] == 1],
        "sarkazm": int(sark[i]),
    })
with open(RESULTS / "interp_sentences.json", "w") as f:
    json.dump(sel_records, f, ensure_ascii=False, indent=2)
print(f"[save] {RESULTS / 'interp_sentences.json'}  ({len(sel_records)} sentences)")
for r in sel_records:
    print(f"   [{r['tag']}] idx={r['test_index']} true={r['true']} :: {r['tekst'][:80]!r}")

# ---------------------------------------------------------------- local char attribution
_ws = re.compile(r"\s\s+")

def char_attributions(text: str, e_idx: int) -> tuple[str, np.ndarray]:
    """Per-character contribution toward emotion e_idx (exact additive decomposition of
    the linear model: coef*tfidf per feature), distributed over char positions of each
    covering char_wb n-gram. Returns the preprocessed (lowercased, ws-collapsed) string
    and the per-char attribution array."""
    pre = _ws.sub(" ", text.lower())
    row = vec.transform([text])
    vocab = vec.vocabulary_
    cval = {idx: coef[e_idx, idx] * v for idx, v in zip(row.indices, row.data)}
    attr = np.zeros(len(pre))
    off = 0
    for word in pre.split(" "):
        if word:
            padded = " " + word + " "
            for n in range(3, 6):
                for o in range(0, len(padded) - n + 1):
                    ng = padded[o:o + n]
                    idx = vocab.get(ng)
                    if idx is None or idx not in cval:
                        continue
                    share = cval[idx] / n
                    for k in range(n):
                        p = o + k
                        if 1 <= p <= len(word):
                            attr[off + (p - 1)] += share
        off += len(word) + 1  # word + the single separating space
    return pre, attr


def render_heatmap(pre: str, attr: np.ndarray, title: str, path: Path, width: int = 60) -> None:
    if np.allclose(attr, 0):
        amax = 1.0
    else:
        amax = np.abs(attr).max()
    norm = TwoSlopeNorm(vmin=-amax, vcenter=0.0, vmax=amax)
    cmap = plt.get_cmap("RdBu_r")
    rows_chars = [pre[i:i + width] for i in range(0, len(pre), width)]
    rows_attr = [attr[i:i + width] for i in range(0, len(attr), width)]
    fig, ax = plt.subplots(figsize=(0.16 * width + 1, 0.5 * len(rows_chars) + 1.2))
    ax.set_xlim(0, width)
    ax.set_ylim(0, len(rows_chars))
    ax.axis("off")
    for r, (chars, avals) in enumerate(zip(rows_chars, rows_attr)):
        y = len(rows_chars) - r - 1
        for c, (ch, a) in enumerate(zip(chars, avals)):
            ax.add_patch(plt.Rectangle((c, y), 1, 0.9, color=cmap(norm(a)), ec="none"))
            ax.text(c + 0.5, y + 0.45, ch, ha="center", va="center",
                    family="monospace", fontsize=11)
    ax.set_title(title, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


for r in sel_records:
    i = r["test_index"]
    # attribute toward the emotions the model predicts (fallback: top proba)
    emos = r["classical_pred"] or [EMOTIONS[int(proba_test[i].argmax())]]
    for emo in emos[:2]:
        e_idx = EMOTIONS.index(emo)
        pre, attr = char_attributions(r["tekst"], e_idx)
        title = f"[{r['tag']}] → {emo}  (czerwony = za, niebieski = przeciw)"
        out = FIGURES / f"interp_classical_{r['tag']}_{emo}.pdf"
        render_heatmap(pre, attr, title, out)
        print(f"[save] {out}")

print("\n[done] Part A complete.")
