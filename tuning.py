"""Shared plumbing for tuning and evaluation: feature matrices, the two `age`
encodings, threshold selection, and the metric set the proposal committed to.

Kept separate from decision_tree.py so the appendix listing of the algorithm
itself stays free of experiment scaffolding.
"""

import numpy as np
import pandas as pd

from decision_tree import NUMERIC_COLS

TARGET = "readmitted_binary"

# `age` is recorded as ten ordered buckets. Treated as nominal, a one-vs-rest
# split can only ask "is age exactly [70-80)?" and never "is age under 70",
# which is the more useful clinical question. Mapping to bucket midpoints buys
# the ordered question at the cost of asserting a linear scale on a variable
# that was never measured on one. Both encodings are run; see the report.
AGE_MIDPOINT = {f"[{lo}-{lo + 10})": lo + 5 for lo in range(0, 100, 10)}


def feature_columns(df, age_numeric):
    """Column lists for one age encoding. Order is fixed so matrices built from
    different frames stay aligned."""
    numeric = list(NUMERIC_COLS)
    categorical = [c for c in df.columns if c not in NUMERIC_COLS + [TARGET]]
    if age_numeric:
        categorical.remove("age")
        numeric.append("age")
    return numeric, categorical


def matrices(df, numeric, categorical):
    """(X_num, X_cat, y) for a frame, using column lists from feature_columns."""
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
    """Precision, recall and F1 for the positive (readmitted <30) class.

    Written out rather than imported so the report can show the arithmetic
    behind the numbers it quotes; sklearn's versions are used as a cross-check
    in benchmark.py.
    """
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
    """Precision/recall/F1 at every threshold the model can actually produce.

    A tree emits one probability per leaf, so the distinct leaf probabilities
    are the complete candidate set -- sweeping a fixed grid would either miss
    achievable operating points or waste work on identical ones.
    """
    rows = []
    for t in np.unique(proba):
        p, r, f = prf(y_true, (proba >= t).astype(int))
        rows.append({"threshold": float(t), "precision": p, "recall": r, "f1": f})
    return pd.DataFrame(rows)


def best_threshold(y_true, proba):
    """Threshold maximising positive-class F1. Ties break to the higher
    threshold, i.e. the more conservative model."""
    sweep = threshold_sweep(y_true, proba)
    best = sweep.loc[sweep["f1"].idxmax()]
    return float(best["threshold"]), float(best["f1"])
