from dataclasses import dataclass

import numpy as np


def gini(y):
    """Gini impurity of a 0/1 label array."""
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y, minlength=2)
    proportions = counts / len(y)
    return 1.0 - np.sum(proportions ** 2)


def weighted_impurity(y_left, y_right):
    """Child impurities weighted by child size."""
    n = len(y_left) + len(y_right)
    if n == 0:
        return 0.0
    return (len(y_left) * gini(y_left) + len(y_right) * gini(y_right)) / n


def impurity_decrease(y, y_left, y_right):
    """Gain of a split over its parent."""
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
# the three *_id columns are category codes, so they are not listed here


def best_numeric_split(x, y, min_samples_leaf=1):
    """Best threshold for one numeric feature: (threshold, gain), or (None, 0.0)."""
    n = len(y)
    if n < 2:
        return None, 0.0

    order = np.argsort(x, kind="mergesort")  # stable, so ties are reproducible
    x_sorted = x[order]
    y_sorted = y[order]

    # cut i sits between sorted rows i and i + 1
    n_left = np.arange(1, n)
    n_right = n - n_left
    pos_left = np.cumsum(y_sorted)[:-1]
    pos_right = y_sorted.sum() - pos_left

    # binary Gini is 2p(1 - p)
    p_left = pos_left / n_left
    p_right = pos_right / n_right
    weighted = (n_left * 2 * p_left * (1 - p_left)
                + n_right * 2 * p_right * (1 - p_right)) / n

    # skip cuts between equal values or that break min_samples_leaf
    valid = ((x_sorted[:-1] < x_sorted[1:])
             & (n_left >= min_samples_leaf)
             & (n_right >= min_samples_leaf))
    if not valid.any():
        return None, 0.0
    weighted = np.where(valid, weighted, np.inf)

    i = int(np.argmin(weighted))
    threshold = (x_sorted[i] + x_sorted[i + 1]) / 2.0
    return threshold, gini(y) - weighted[i]


def best_numeric_feature(X, y, feature_names, min_samples_leaf=1):
    """Best split over the numeric features: ((index, name, threshold), gain)."""
    best, best_gain = None, 0.0
    for j, name in enumerate(feature_names):
        threshold, gain = best_numeric_split(X[:, j], y, min_samples_leaf)
        if threshold is not None and gain > best_gain:
            best, best_gain = (j, name, threshold), gain
    return best, best_gain


@dataclass
class Split:
    """One binary test at an internal node."""

    kind: str           # "numeric" or "categorical"
    feature: int        # column index
    name: str
    value: object       # threshold or category
    gain: float


def best_categorical_split(x, y, min_samples_leaf=1):
    """Best one-vs-rest split (x == value): (value, gain), or (None, 0.0)."""
    values, codes = np.unique(x, return_inverse=True)
    k = len(values)
    if k < 2:
        return None, 0.0

    n = len(y)
    n_in = np.bincount(codes, minlength=k)
    pos_in = np.bincount(codes, weights=y, minlength=k)  # positives per category
    n_out = n - n_in
    pos_out = y.sum() - pos_in

    p_in = pos_in / n_in
    p_out = pos_out / n_out
    weighted = (n_in * 2 * p_in * (1 - p_in)
                + n_out * 2 * p_out * (1 - p_out)) / n

    # skip categories that break min_samples_leaf
    valid = (n_in >= min_samples_leaf) & (n_out >= min_samples_leaf)
    if not valid.any():
        return None, 0.0
    weighted = np.where(valid, weighted, np.inf)

    i = int(np.argmin(weighted))
    return values[i], gini(y) - weighted[i]


def best_split(X_num, X_cat, y, numeric_names, categorical_names,
               min_samples_leaf=1):
    """Best split across all features, or None if nothing has positive gain."""
    best = None

    for j, name in enumerate(numeric_names):
        threshold, gain = best_numeric_split(X_num[:, j], y, min_samples_leaf)
        if threshold is not None and (best is None or gain > best.gain):
            best = Split("numeric", j, name, threshold, gain)

    for j, name in enumerate(categorical_names):
        value, gain = best_categorical_split(X_cat[:, j], y, min_samples_leaf)
        if value is not None and (best is None or gain > best.gain):
            best = Split("categorical", j, name, value, gain)

    return best if best is not None and best.gain > 0 else None


def apply_split(split, X_num, X_cat):
    """Mask of rows that go to the left child."""
    if split.kind == "numeric":
        return X_num[:, split.feature] <= split.value
    return X_cat[:, split.feature] == split.value


@dataclass
class Node:
    """Tree node. Stores class counts, not a label, so leaves give probabilities."""

    counts: np.ndarray          # [n_negative, n_positive]
    split: "Split | None" = None
    left: "Node | None" = None
    right: "Node | None" = None

    @property
    def is_leaf(self):
        return self.split is None


def build(X_num, X_cat, y, numeric_names, categorical_names,
          max_depth=None, min_samples_split=2, min_samples_leaf=1, depth=0):
    """Grow the tree recursively and return the root."""
    counts = np.bincount(y, minlength=2)
    node = Node(counts=counts)

    if counts[0] == 0 or counts[1] == 0:
        return node
    if max_depth is not None and depth >= max_depth:
        return node
    if len(y) < min_samples_split:
        return node

    split = best_split(X_num, X_cat, y, numeric_names, categorical_names,
                       min_samples_leaf)
    if split is None:
        return node

    mask = apply_split(split, X_num, X_cat)
    node.split = split
    node.left = build(X_num[mask], X_cat[mask], y[mask],
                      numeric_names, categorical_names,
                      max_depth, min_samples_split, min_samples_leaf, depth + 1)
    node.right = build(X_num[~mask], X_cat[~mask], y[~mask],
                       numeric_names, categorical_names,
                       max_depth, min_samples_split, min_samples_leaf, depth + 1)
    return node


def _fill_proba(node, X_num, X_cat, idx, out):
    """Send a block of rows down the tree and write leaf probabilities to out."""
    if node.is_leaf:
        total = node.counts.sum()
        out[idx] = node.counts[1] / total if total else 0.0
        return
    mask = apply_split(node.split, X_num, X_cat)
    _fill_proba(node.left, X_num[mask], X_cat[mask], idx[mask], out)
    _fill_proba(node.right, X_num[~mask], X_cat[~mask], idx[~mask], out)


def predict_proba(node, X_num, X_cat):
    """P(readmitted within 30 days) for each row."""
    n = X_num.shape[0]
    out = np.empty(n, dtype=float)
    _fill_proba(node, X_num, X_cat, np.arange(n), out)
    return out


def predict(node, X_num, X_cat, threshold=0.5):
    """0/1 predictions at the given threshold."""
    return (predict_proba(node, X_num, X_cat) >= threshold).astype(int)


def tree_depth(node):
    """Longest root-to-leaf path. A lone leaf has depth 0."""
    if node.is_leaf:
        return 0
    return 1 + max(tree_depth(node.left), tree_depth(node.right))


def count_nodes(node):
    """Internal plus leaf nodes, like sklearn's tree_.node_count."""
    if node.is_leaf:
        return 1
    return 1 + count_nodes(node.left) + count_nodes(node.right)


def count_leaves(node):
    if node.is_leaf:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)


def feature_importances(node, normalise=True):
    """Gain per feature, weighted by the share of rows reaching each split
    (sklearn's definition). Normalised to sum to 1."""
    total = node.counts.sum()
    scores = {}

    def walk(n):
        if n.is_leaf:
            return
        share = n.counts.sum() / total
        scores[n.split.name] = scores.get(n.split.name, 0.0) + share * n.split.gain
        walk(n.left)
        walk(n.right)

    walk(node)
    if normalise:
        s = sum(scores.values())
        if s:
            scores = {k: v / s for k, v in scores.items()}
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def describe(node):
    """One-line text for a node: its test, or its leaf rate."""
    n = int(node.counts.sum())
    rate = node.counts[1] / n if n else 0.0
    if node.is_leaf:
        return f"leaf  n={n:,}  {100 * rate:.1f}% readmitted"
    test = (f"{node.split.name} <= {node.split.value:g}"
            if node.split.kind == "numeric"
            else f"{node.split.name} == {node.split.value}")
    return f"{test}  (gain {node.split.gain:.5f}, n={n:,}, {100 * rate:.1f}% readmitted)"


def render_tree(node, max_depth=3, _prefix="", _depth=0, _lines=None):
    """Text render of the top levels. The left branch is the True side."""
    if _lines is None:
        _lines = [describe(node)]
    if node.is_leaf or _depth >= max_depth:
        return "\n".join(_lines)
    for child, is_last, label in ((node.left, False, "T"), (node.right, True, "F")):
        elbow = "`--" if is_last else "|--"
        _lines.append(f"{_prefix}{elbow} {label}: {describe(child)}")
        render_tree(child, max_depth, _prefix + ("    " if is_last else "|   "),
                    _depth + 1, _lines)
    return "\n".join(_lines)
