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


def best_numeric_split(x, y, min_samples_leaf=1):
    """Best threshold for a single numeric feature.

    Returns (threshold, impurity_decrease), or (None, 0.0) when no valid split
    exists (fewer than two rows, every value identical, or no cut leaving
    min_samples_leaf rows on both sides).
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

    # A cut between two identical values is not a real split, and a cut leaving
    # fewer than min_samples_leaf rows on a side is not a legal one. Both are
    # excluded here, inside the search, rather than by rejecting the winner
    # afterwards -- that would turn a node into a leaf when a legal split existed.
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
    """Search every numeric feature. Returns ((index, name, threshold), gain)."""
    best, best_gain = None, 0.0
    for j, name in enumerate(feature_names):
        threshold, gain = best_numeric_split(X[:, j], y, min_samples_leaf)
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


def best_categorical_split(x, y, min_samples_leaf=1):
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

    # same reasoning as the numeric sweep: a category too small to be its own
    # leaf is not a candidate, so it is excluded before the argmin
    valid = (n_in >= min_samples_leaf) & (n_out >= min_samples_leaf)
    if not valid.any():
        return None, 0.0
    weighted = np.where(valid, weighted, np.inf)

    i = int(np.argmin(weighted))
    return values[i], gini(y) - weighted[i]


def best_split(X_num, X_cat, y, numeric_names, categorical_names,
               min_samples_leaf=1):
    """Search every feature of both kinds. Returns the winning Split, or None
    when no split yields a positive gain."""
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
    """Boolean mask over rows: True goes to the left child."""
    if split.kind == "numeric":
        return X_num[:, split.feature] <= split.value
    return X_cat[:, split.feature] == split.value


@dataclass
class Node:
    """One node of the tree.

    `counts` is the class distribution of the training rows that reached this
    node, not a majority label. Storing the distribution is what makes leaf
    probabilities and a tuned decision threshold possible later -- at a 9.10%
    positive rate a majority-vote leaf predicts "not readmitted" almost
    everywhere, so the label alone would throw away the only information the
    imbalance leaves us.
    """

    counts: np.ndarray          # [n_negative, n_positive]
    split: "Split | None" = None
    left: "Node | None" = None
    right: "Node | None" = None

    @property
    def is_leaf(self):
        return self.split is None


def build(X_num, X_cat, y, numeric_names, categorical_names,
          max_depth=None, min_samples_split=2, min_samples_leaf=1, depth=0):
    """Recursively grow a binary tree. Returns the root Node.

    Stops when the node is pure, the depth limit is hit, there are too few rows
    to split, or no split yields a positive impurity decrease. The
    min_samples_leaf rule is enforced inside the split search rather than here.
    """
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
    """Push a block of rows down one node, splitting the block by the mask.

    Rows travel in groups rather than one at a time: every row in a block takes
    the same test, so one vectorised comparison routes the whole block. A
    per-row walk in Python is fine once, but not across a depth sweep times a
    threshold sweep.
    """
    if node.is_leaf:
        total = node.counts.sum()
        out[idx] = node.counts[1] / total if total else 0.0
        return
    mask = apply_split(node.split, X_num, X_cat)
    _fill_proba(node.left, X_num[mask], X_cat[mask], idx[mask], out)
    _fill_proba(node.right, X_num[~mask], X_cat[~mask], idx[~mask], out)


def predict_proba(node, X_num, X_cat):
    """P(readmitted within 30 days) per row, from the reached leaf's counts."""
    n = X_num.shape[0]
    out = np.empty(n, dtype=float)
    _fill_proba(node, X_num, X_cat, np.arange(n), out)
    return out


def predict(node, X_num, X_cat, threshold=0.5):
    """Hard 0/1 prediction. The threshold is tuned on validation, not left at
    0.5 -- see the imbalance discussion in the report."""
    return (predict_proba(node, X_num, X_cat) >= threshold).astype(int)


def tree_depth(node):
    """Longest root-to-leaf path. A single-node tree has depth 0."""
    if node.is_leaf:
        return 0
    return 1 + max(tree_depth(node.left), tree_depth(node.right))


def count_nodes(node):
    """Total nodes, internal and leaf -- comparable to sklearn's tree_.node_count."""
    if node.is_leaf:
        return 1
    return 1 + count_nodes(node.left) + count_nodes(node.right)


def count_leaves(node):
    if node.is_leaf:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)


def feature_importances(node, normalise=True):
    """Total impurity decrease attributable to each feature, weighted by how
    many training rows reached the splitting node.

    This is the same definition sklearn uses for `feature_importances_`: a split
    counts for more when it is made high in the tree where it separates many
    rows, so a weak split at the root can outrank a strong one in a small
    branch. `Split.gain` is already the parent impurity minus the size-weighted
    child impurity, so the node's contribution is just that gain scaled by its
    share of the training set.

    Returned normalised to sum to 1, which is what makes the numbers comparable
    across models rather than across-the-board larger for a deeper tree.
    """
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
    """One-line description of a node: its test, or its leaf distribution."""
    n = int(node.counts.sum())
    rate = node.counts[1] / n if n else 0.0
    if node.is_leaf:
        return f"leaf  n={n:,}  {100 * rate:.1f}% readmitted"
    test = (f"{node.split.name} <= {node.split.value:g}"
            if node.split.kind == "numeric"
            else f"{node.split.name} == {node.split.value}")
    return f"{test}  (gain {node.split.gain:.5f}, n={n:,}, {100 * rate:.1f}% readmitted)"


def render_tree(node, max_depth=3, _prefix="", _depth=0, _lines=None):
    """Plain-text render of the top of the tree.

    Only the first few levels are legible on a page -- the tuned tree has 151
    nodes -- and the top levels are the ones carrying most of the signal anyway.
    Left branch is the True side of the test.
    """
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
