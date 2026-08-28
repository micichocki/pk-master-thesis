"""Regenerate the representation x classifier grid heatmap (Fig. 5.1)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results" / "exp1_grid_twitteremo.csv"
FIGURES_DIR = ROOT / "thesis" / "images"

#: Display names kept identical to the summary table.
REPRESENTATION_LABELS: dict[str, str] = {
    "tfidf_char": "TF-IDF znakowy",
    "tfidf_wordchar": "TF-IDF słowny+znakowy",
    "combined": "połączona",
    "tfidf_word": "TF-IDF słowny",
    "fasttext": "fastText",
    "lsa": "LSA",
    "stats": "statystyki",
    "nrc": "NRC (leksykon)",
    "lda": "LDA",
}

MODEL_LABELS: dict[str, str] = {
    "logreg": "LogReg",
    "ridge": "Ridge",
    "linearsvc": "LinearSVC",
    "complement_nb": "ComplementNB",
    "random_forest": "RandomForest",
    "extra_trees": "ExtraTrees",
    "lightgbm": "LightGBM",
    "hist_gb": "HistGB",
    "mlp": "MLP",
    "knn": "$k$-NN",
}

ANNOT_SIZE = 15
TICK_SIZE = 16
TITLE_SIZE = 19
CBAR_SIZE = 15


def build_figure(grid: pd.DataFrame) -> plt.Figure:
    """Render the two stacked heatmaps for the grid of experiment 1.

    Args:
        grid: Long-format results with ``model``, ``representation``,
            ``f1_macro`` and ``time_s`` columns.

    Returns:
        The assembled matplotlib figure.
    """
    grid = grid.assign(
        representation=grid["representation"].map(REPRESENTATION_LABELS),
        model=grid["model"].map(MODEL_LABELS),
    )

    # Order both axes by best achieved F1-Macro so the strong corner is top-left.
    rep_order = (
        grid.groupby("representation")["f1_macro"].max().sort_values(ascending=False).index
    )
    model_order = grid.groupby("model")["f1_macro"].max().sort_values(ascending=False).index

    pivot_f1 = grid.pivot(index="model", columns="representation", values="f1_macro")
    pivot_time = grid.pivot(index="model", columns="representation", values="time_s")
    pivot_f1 = pivot_f1.loc[model_order, rep_order]
    pivot_time = pivot_time.loc[model_order, rep_order]

    fig, axes = plt.subplots(2, 1, figsize=(11, 15))

    sns.heatmap(
        pivot_f1,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        ax=axes[0],
        linewidths=0.5,
        annot_kws={"size": ANNOT_SIZE},
        cbar_kws={"label": "F1-Macro"},
    )
    axes[0].set_title("F1-Macro (TwitterEmo, zbiór walidacyjny)", fontsize=TITLE_SIZE, pad=14)

    sns.heatmap(
        pivot_time,
        annot=True,
        fmt=".1f",
        cmap="OrRd",
        ax=axes[1],
        linewidths=0.5,
        annot_kws={"size": ANNOT_SIZE},
        cbar_kws={"label": "Czas uczenia [s]"},
    )
    axes[1].set_title("Czas uczenia [s]", fontsize=TITLE_SIZE, pad=14)

    for ax in axes:
        ax.set_xlabel("Reprezentacja", fontsize=TICK_SIZE)
        ax.set_ylabel("Model", fontsize=TICK_SIZE)
        ax.tick_params(axis="x", labelsize=TICK_SIZE, rotation=40)
        ax.tick_params(axis="y", labelsize=TICK_SIZE, rotation=0)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=CBAR_SIZE)
        cbar.set_label(cbar.ax.get_ylabel(), size=CBAR_SIZE)

    fig.tight_layout(h_pad=3.0)
    return fig


def main() -> None:
    """Write the regenerated figure to ``figures/``."""
    grid = pd.read_csv(RESULTS)
    fig = build_figure(grid)
    out = FIGURES_DIR / "exp1_grid_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"zapisano {out}")


if __name__ == "__main__":
    main()
