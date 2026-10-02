"""Shared helpers for tuning and evaluation: feature matrices, age encodings,
metrics and the threshold sweep."""

import numpy as np
import pandas as pd

from decision_tree import NUMERIC_COLS

TARGET = "readmitted_binary"

# age bucket -> midpoint, so the numeric encoding can split on "under 70"
AGE_MIDPOINT = {f"[{lo}-{lo + 10})": lo + 5 for lo in range(0, 100, 10)}


def feature_columns(df, age_numeric):
    """Numeric and categorical column lists for one age encoding."""
    numeric = list(NUMERIC_COLS)
    categorical = [c for c in df.columns if c not in NUMERIC_COLS + [TARGET]]
    if age_numeric:
        categorical.remove("age")
        numeric.append("age")
    return numeric, categorical


def matrices(df, numeric, categorical):
    """(X_num, X_cat, y) for a frame."""
    d = df
    if "age" in numeric:
        d = df.copy()
        d["age"] = d["age"].map(AGE_MIDPOINT)
        assert d["age"].notna().all(), "unmapped age bucket"
    X_num = d[numeric].to_numpy(dtype=float)
    X_cat = d[categorical].astype(str).to_numpy()
    y = d[TARGET].to_numpy(dtype=int)
    return X_num, X_cat, y


def prf(y_true, y_pred):
    """Precision, recall and F1 for the positive class."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def confusion(y_true, y_pred):
    """(tn, fp, fn, tp)."""
    return (int(((y_pred == 0) & (y_true == 0)).sum()),
            int(((y_pred == 1) & (y_true == 0)).sum()),
            int(((y_pred == 0) & (y_true == 1)).sum()),
            int(((y_pred == 1) & (y_true == 1)).sum()))


def threshold_sweep(y_true, proba):
    """Precision, recall and F1 at each distinct leaf probability."""
    rows = []
    for t in np.unique(proba):
        p, r, f = prf(y_true, (proba >= t).astype(int))
        rows.append({"threshold": float(t), "precision": p, "recall": r, "f1": f})
    return pd.DataFrame(rows)


def best_threshold(y_true, proba):
    """Threshold with the highest F1. Ties go to the lowest threshold."""
    sweep = threshold_sweep(y_true, proba)
    best = sweep.loc[sweep["f1"].idxmax()]
    return float(best["threshold"]), float(best["f1"])
