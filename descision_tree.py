from dataclasses import dataclass

import numpy as np


def gini(y):
    """Gini impurity of a label array.

    0.0 when every label is the same, 0.5 at a 50/50 binary split. Cheaper than
    entropy (no logs) and is what sklearn uses by default, which keeps the
    benchmark comparison like-for-like.
    """
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y, minlength=2)
    proportions = counts / len(y)
    return 1.0 - np.sum(proportions ** 2)


def weighted_impurity(y_left, y_right):
    """Impurity of a proposed split: child impurities weighted by child size.

    Weighting by size is what stops the search preferring a split that carves off
    three pure rows and leaves the other 50,000 untouched.
    """
    n = len(y_left) + len(y_right)
    if n == 0:
        return 0.0
    return (len(y_left) * gini(y_left) + len(y_right) * gini(y_right)) / n


def impurity_decrease(y, y_left, y_right):
    """How much a split improves on the parent. The split search maximises this."""
    return gini(y) - weighted_impurity(y_left, y_right)




NUMERIC_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]
# note: admission_type_id / discharge_disposition_id / admission_source_id are
# integer-coded *categories*, not numbers. They are deliberately not in this list.


def best_numeric_split(x, y):
    """Best threshold for a single numeric feature.

    Returns (threshold, impurity_decrease), or (None, 0.0) when no valid split
    exists (fewer than two rows, or every value identical).
    """
    n = len(y)
    if n < 2:
        return None, 0.0

    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    y_sorted = y[order]

    # every possible cut sits between position i and i+1, for i = 0 .. n-2
    n_left = np.arange(1, n)
    n_right = n - n_left
    pos_left = np.cumsum(y_sorted)[:-1]
    pos_right = y_sorted.sum() - pos_left

    # for a binary target, Gini from a positive proportion p is just 2p(1-p)
    p_left = pos_left / n_left
    p_right = pos_right / n_right
    weighted = (n_left * 2 * p_left * (1 - p_left)
                + n_right * 2 * p_right * (1 - p_right)) / n

    # a cut between two identical values is not a real split
    valid = x_sorted[:-1] < x_sorted[1:]
    if not valid.any():
        return None, 0.0
    weighted = np.where(valid, weighted, np.inf)

    i = int(np.argmin(weighted))
    threshold = (x_sorted[i] + x_sorted[i + 1]) / 2.0
    return threshold, gini(y) - weighted[i]


def best_numeric_feature(X, y, feature_names):
    """Search every numeric feature. Returns ((index, name, threshold), gain)."""
    best, best_gain = None, 0.0
    for j, name in enumerate(feature_names):
        threshold, gain = best_numeric_split(X[:, j], y)
        if threshold is not None and gain > best_gain:
            best, best_gain = (j, name, threshold), gain
    return best, best_gain


@dataclass
class Split:
    """A single binary test at an internal node."""

    kind: str           # "numeric" or "categorical"
    feature: int        # column index within its own array
    name: str
    value: object       # threshold (numeric) or category (categorical)
    gain: float


def best_categorical_split(x, y):
    """Best one-vs-rest split for a single categorical feature.

    Splits are `x == value` vs everything else, which keeps every node binary and
    matches what sklearn does on a one-hot encoded column. The theoretically
    optimal split would consider all 2^(k-1)-1 subsets of the k categories;
    with medical_specialty at 67 distinct values that is not tractable, and
    one-vs-rest costs k evaluations instead.
    """
    values, codes = np.unique(x, return_inverse=True)
    k = len(values)
    if k < 2:
        return None, 0.0

    n = len(y)
    # summing the 0/1 target within each bucket counts positives per category
    n_in = np.bincount(codes, minlength=k)
    pos_in = np.bincount(codes, weights=y, minlength=k)
    n_out = n - n_in
    pos_out = y.sum() - pos_in

    p_in = pos_in / n_in
    p_out = pos_out / n_out
    weighted = (n_in * 2 * p_in * (1 - p_in)
                + n_out * 2 * p_out * (1 - p_out)) / n

    i = int(np.argmin(weighted))
    return values[i], gini(y) - weighted[i]


def best_split(X_num, X_cat, y, numeric_names, categorical_names):
    """Search every feature of both kinds. Returns the winning Split, or None
    when no split yields a positive gain."""
    best = None

    for j, name in enumerate(numeric_names):
        threshold, gain = best_numeric_split(X_num[:, j], y)
        if threshold is not None and (best is None or gain > best.gain):
            best = Split("numeric", j, name, threshold, gain)

    for j, name in enumerate(categorical_names):
        value, gain = best_categorical_split(X_cat[:, j], y)
        if value is not None and (best is None or gain > best.gain):
            best = Split("categorical", j, name, value, gain)

    return best if best is not None and best.gain > 0 else None


def apply_split(split, X_num, X_cat):
    """Boolean mask over rows: True goes to the left child."""
    if split.kind == "numeric":
        return X_num[:, split.feature] <= split.value
    return X_cat[:, split.feature] == split.value
