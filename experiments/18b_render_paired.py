"""Interpretability — render paired attribution figures.

Run from repo root (after the kernel output is downloaded):
  .venv/bin/python experiments/18b_render_paired.py
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "thesis" / "images"
IG_JSON = ROOT / "kaggle" / "herbert_large_ig" / "output" / "herbert_ig.json"

with open(ROOT / "data" / "features" / "interp_classical_model.pkl", "rb") as f:
    M = pickle.load(f)
vec, coef, EMOTIONS = M["vectorizer"], M["coef"], M["emotions"]
_ws = re.compile(r"\s\s+")

TAGS = {646: "single_radość", 1290: "single_wstręt", 556: "single_strach",
        1362: "single_przeczuwanie", 1207: "sarkazm", 985: "multi"}

# Dopełniacz emocji — tytuł rysunku brzmi „dla przykładu z emocją strachu", więc
# nazwy trzeba odmienić; mianownik zostaje jako zabezpieczenie na wypadek nowej etykiety.
GENITIVE = {"radość": "radości", "smutek": "smutku", "zaufanie": "zaufania",
            "wstręt": "wstrętu", "strach": "strachu", "gniew": "gniewu",
            "przeczuwanie": "przeczuwania", "zdziwienie": "zdziwienia"}


def char_attributions(text: str, e_idx: int) -> tuple[str, np.ndarray]:
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
                    idx = vocab.get(padded[o:o + n])
                    if idx is None or idx not in cval:
                        continue
                    share = cval[idx] / n
                    for k in range(n):
                        p = o + k
                        if 1 <= p <= len(word):
                            attr[off + (p - 1)] += share
        off += len(word) + 1
    return pre, attr


def clean_token(t: str) -> str:
    return t.replace("</w>", "").replace("▁", "").replace("##", "").replace("Ġ", "")


def draw_char_row(ax, pre, attr, width=70):
    amax = np.abs(attr).max() or 1.0
    norm = TwoSlopeNorm(vmin=-amax, vcenter=0.0, vmax=amax)
    cmap = plt.get_cmap("RdBu_r")
    rows = [(pre[i:i + width], attr[i:i + width]) for i in range(0, len(pre), width)]
    ax.set_xlim(0, width)
    ax.set_ylim(0, max(len(rows), 1))
    ax.axis("off")
    for r, (chars, avals) in enumerate(rows):
        y = len(rows) - r - 1
        for c, (ch, a) in enumerate(zip(chars, avals)):
            ax.add_patch(plt.Rectangle((c, y), 1, 0.9, color=cmap(norm(a)), ec="none"))
            ax.text(c + 0.5, y + 0.45, ch, ha="center", va="center", family="monospace", fontsize=9)


def draw_token_row(ax, tokens, attr, width=70):
    disp = [clean_token(t) for t in tokens]
    attr = np.asarray(attr)
    amax = np.abs(attr).max() or 1.0
    norm = TwoSlopeNorm(vmin=-amax, vcenter=0.0, vmax=amax)
    cmap = plt.get_cmap("RdBu_r")
    # greedy wrap by cumulative char width (+1 padding per token)
    rows, cur, curw = [], [], 0
    for tk, a in zip(disp, attr):
        w = max(len(tk), 1) + 1
        if curw + w > width and cur:
            rows.append(cur); cur, curw = [], 0
        cur.append((tk, a, w)); curw += w
    if cur:
        rows.append(cur)
    ax.set_xlim(0, width)
    ax.set_ylim(0, max(len(rows), 1))
    ax.axis("off")
    for r, row in enumerate(rows):
        y = len(rows) - r - 1
        x = 0
        for tk, a, w in row:
            ax.add_patch(plt.Rectangle((x, y), w - 0.2, 0.9, color=cmap(norm(a)), ec="none"))
            ax.text(x + (w - 0.2) / 2, y + 0.45, tk, ha="center", va="center",
                    family="monospace", fontsize=9)
            x += w


def main():
    records = json.load(open(IG_JSON))
    by_key = {(r["test_index"], r["emotion"]): r for r in records}
    for (idx, emo), r in by_key.items():
        tag = TAGS.get(idx, str(idx))
        pre, cattr = char_attributions(r["tekst"], EMOTIONS.index(emo))
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 4.2),
                                     gridspec_kw={"height_ratios": [1, 1]})
        draw_token_row(a1, r["tokens"], r["attributions"])
        # bez [proba=...]: prawdopodobieństwo modelu nie ma związku z atrybucją,
        # a w podpisie rysunku wygląda na liczbę, którą trzeba interpretować
        a1.set_title(f"HerBERT-large (integrated gradients) → {emo}",
                     fontsize=11, loc="left")
        draw_char_row(a2, pre, cattr)
        a2.set_title("LogReg + TF-IDF char (dekompozycja liniowa coef·TF-IDF, per-znak)",
                     fontsize=11, loc="left")
        # bez wewnętrznego znacznika [tag] — to identyfikator z kodu, nie nazwa dla
        # czytelnika; „przykład", nie „zdanie", bo teksty są tweetami (średnio 1,75 zdania)
        fig.suptitle(f"Atrybucja dla przykładu z emocją {GENITIVE.get(emo, emo)}"
                     f" (czerwony = za emocją, niebieski = przeciw)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = FIGURES / f"interp_paired_{tag}_{emo}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] {out}")


if __name__ == "__main__":
    main()
