"""Shared helpers for the local notebooks and scripts: metrics, per-label threshold
tuning, bootstrap CIs, data loading, the weighted multi-label Trainer, MLflow logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, hamming_loss, jaccard_score, accuracy_score,
    precision_score, recall_score,
)

# Plutchik's 8 emotions, in the canonical order used by every experiment.
EMOTIONS = ["radość", "smutek", "zaufanie", "wstręt", "strach",
            "gniew", "przeczuwanie", "zdziwienie"]
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Standard multi-label metric bundle (all with ``zero_division=0``)."""
    return {
        "f1_macro":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro":        f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_weighted":     f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "hamming_loss":    hamming_loss(y_true, y_pred),
        "jaccard_macro":   jaccard_score(y_true, y_pred, average="macro", zero_division=0),
        "subset_accuracy": accuracy_score(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Per-label decision thresholds (tuned on validation, never on test)
# --------------------------------------------------------------------------- #
def optimal_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    lo: float = 0.05,
    hi: float = 0.95,
    step: float = 0.01,
    default: float = 0.5,
) -> np.ndarray:
    """Per-label threshold maximising F1, searched on a grid.

    The number of labels is inferred from ``y_true.shape[1]``. ``default`` is the
    fallback threshold for a label whose best F1 over the grid is 0 (never
    improves on the initial value).

    The default grid (``0.05..0.95`` step ``0.01``) matches every experiment
    except ``08_lexicon_baseline``, which used ``lo=0.0, hi=0.95, step=0.005``.
    """
    n_labels = y_true.shape[1]
    thresholds = np.full(n_labels, default, dtype=float)
    for i in range(n_labels):
        best_f1, best_t = 0.0, default
        for t in np.arange(lo, hi, step):
            pred = (y_proba[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[i] = best_t
    return thresholds


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals (resampling the test set)
# --------------------------------------------------------------------------- #
def bootstrap_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    seed: int = RANDOM_STATE,
) -> tuple[float, float, float]:
    """Percentile bootstrap 95% CI for F1-Macro. Returns ``(base, lo, hi)``."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    base = f1_score(y_true, y_pred, average="macro", zero_division=0)
    boots = [
        f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
        for idx in (rng.integers(0, n, n) for _ in range(n_boot))
    ]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return base, lo, hi


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = RANDOM_STATE,
) -> tuple[float, float, float, float]:
    """Bootstrap distribution of the F1-Macro difference ``B − A`` on a shared
    resample of the test set (a paired test — more sensitive than comparing two
    marginal CIs).

    Returns ``(mean_diff, ci_lo, ci_hi, p_b_better)`` where ``p_b_better`` is the
    fraction of resamples in which B beats A.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        fa = f1_score(y_true[idx], pred_a[idx], average="macro", zero_division=0)
        fb = f1_score(y_true[idx], pred_b[idx], average="macro", zero_division=0)
        diffs[k] = fb - fa
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return diffs.mean(), lo, hi, float((diffs > 0).mean())


# --------------------------------------------------------------------------- #
# Data loading / feature caching
# --------------------------------------------------------------------------- #
def load_splits(
    prefix: str,
    processed_dir: str | Path = "../data/processed",
    text_col: str = "tekst",
    label_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load train/val/test CSVs for a dataset, filling NaN text with ``""``.

    Returns ``{"train": df, "val": df, "test": df}``; each df gets the raw label
    matrix attached as ``df.attrs["y"]`` for convenience.
    """
    processed_dir = Path(processed_dir)
    label_cols = label_cols or EMOTIONS
    out: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        df = pd.read_csv(processed_dir / f"{prefix}_{split}.csv").reset_index(drop=True)
        df[text_col] = df[text_col].fillna("")
        df.attrs["y"] = df[label_cols].values
        out[split] = df
    return out


def cache_features(
    name: str,
    fn: Callable[..., Any],
    *args: Any,
    cache_dir: str | Path = "../data/features",
    force: bool = False,
    **kwargs: Any,
) -> Any:
    """Memoise the result of ``fn(*args, **kwargs)`` to ``cache_dir/name.pkl``."""
    import pickle

    cache_path = Path(cache_dir) / f"{name}.pkl"
    if cache_path.exists() and not force:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    print(f"  Computing {name} (no cache)...")
    result = fn(*args, **kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result


def pos_weight(y_train: np.ndarray, lo: float = 1.0, hi: float = 10.0):
    """Inverse-frequency positive class weights for BCE, clipped to ``[lo, hi]``.

    Returned as a ``torch.FloatTensor`` (torch imported lazily).
    """
    import torch

    pos = y_train.sum(axis=0)
    neg = len(y_train) - pos
    return torch.tensor(np.clip(neg / np.maximum(pos, 1), lo, hi), dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Transformer fine-tuning helpers (torch/transformers imported lazily)
# --------------------------------------------------------------------------- #
def make_weighted_trainer_cls():
    """Return a ``Trainer`` subclass using weighted ``BCEWithLogitsLoss``.

    Factory (not a top-level class) so ``import thesis_lib`` does not require
    transformers. Use:  ``WeightedTrainer = make_weighted_trainer_cls()``.
    """
    import torch.nn.functional as F
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def __init__(self, *args, pos_weight=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.pos_weight = pos_weight

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.binary_cross_entropy_with_logits(
                logits.float(), labels.float(),
                pos_weight=self.pos_weight.to(logits.device)
                if self.pos_weight is not None else None,
            )
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def to_dataset(df: pd.DataFrame, tokenizer, *, text_col: str = "tekst",
               label_cols: list[str] | None = None, max_len: int = 128):
    """Tokenised ``datasets.Dataset`` with float multi-label targets."""
    from datasets import Dataset

    label_cols = label_cols or EMOTIONS
    ds = Dataset.from_dict({
        "text": df[text_col].tolist(),
        "labels": df[label_cols].values.astype("float32").tolist(),
    })
    return ds.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=max_len),
        batched=True, remove_columns=["text"],
    )


# --------------------------------------------------------------------------- #
# MLflow (file-based) logging
# --------------------------------------------------------------------------- #
METRIC_COLS = ["f1_macro", "f1_micro", "f1_weighted", "precision_macro",
               "recall_macro", "hamming_loss", "jaccard_macro", "subset_accuracy"]


def log_runs_from_df(df, phase, name_cols, tag_cols=(), extra_tags=None,
                     metric_cols=METRIC_COLS):
    """Log each row of a results DataFrame as one MLflow run (post-hoc)."""
    import mlflow

    for _, row in df.iterrows():
        run_name = phase + ":" + "__".join(str(row[c]) for c in name_cols)
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("phase", phase)
            for c in tag_cols:
                mlflow.set_tag(c, str(row[c]))
            for k, v in (extra_tags or {}).items():
                mlflow.set_tag(k, str(v))
            mlflow.log_metrics({c: float(row[c]) for c in metric_cols
                                if c in row.index and pd.notna(row[c])})
            if "time_s" in row.index and pd.notna(row["time_s"]):
                mlflow.log_metric("time_s", float(row["time_s"]))
