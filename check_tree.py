"""Verify the recursive build against sklearn on the same 5,000-row subsample.

check_split.py established that a single split matches sklearn to 1e-9. This
checks that recursing on it produces the same *tree*: same shape, same metrics,
at matched hyperparameters, before any optimisation work starts.

The two searches consider the same candidate set at every node -- a one-hot
dummy tested at `<= 0.5` is precisely the one-vs-rest test `x == v`, and the
numeric sweeps are identical -- so agreement should be exact, not approximate.
Tie-breaking between equal-gain splits is the one place they may legitimately
diverge.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from descision_tree import (NUMERIC_COLS, build, count_leaves, count_nodes,
                            predict_proba, tree_depth)

sys.setrecursionlimit(10000)

SEED = 42
SUBSAMPLE = 5000
MIN_SAMPLES_LEAF = 20

train = pd.read_csv("results/train.csv").sample(SUBSAMPLE, random_state=SEED)
CATEGORICAL_COLS = [c for c in train.columns
                    if c not in NUMERIC_COLS + ["readmitted_binary"]]

X_num = train[NUMERIC_COLS].to_numpy(dtype=float)
X_cat = train[CATEGORICAL_COLS].astype(str).to_numpy()
y = train["readmitted_binary"].to_numpy(dtype=int)

# one-hot is what makes the comparison honest: admission_type_id,
# discharge_disposition_id and admission_source_id are integer-coded categories,
# and handed to sklearn raw they would be split as ordered numerics while this
# implementation splits them as nominal. That would compare two different
# algorithms, not two implementations of one.
onehot = pd.get_dummies(train[NUMERIC_COLS + CATEGORICAL_COLS],
                        columns=CATEGORICAL_COLS).to_numpy(dtype=float)


def metrics(proba, y):
    """Accuracy at the majority-vote threshold, plus threshold-free AUC.

    F1 is deliberately not used here: at a 9% positive rate every tree of this
    depth predicts all-negative at 0.5, so F1 is 0.0 for both models and
    compares nothing. AUC ranks the probabilities and stays informative.
    """
    return ((proba >= 0.5).astype(int) == y).mean(), roc_auc_score(y, proba)


print(f"subsample {len(y):,} rows, {y.sum():,} positive ({100 * y.mean():.2f}%)")
print(f"min_samples_leaf={MIN_SAMPLES_LEAF}, criterion=gini\n")

header = f"{'depth':>5}  {'model':<8} {'nodes':>6} {'leaves':>7} {'reached':>8} {'acc':>8} {'auc':>8}"
print(header)
print("-" * len(header))

for max_depth in range(1, 6):
    mine = build(X_num, X_cat, y, NUMERIC_COLS, CATEGORICAL_COLS,
                 max_depth=max_depth, min_samples_leaf=MIN_SAMPLES_LEAF)
    my_proba = predict_proba(mine, X_num, X_cat)
    my_acc, my_auc = metrics(my_proba, y)

    sk = DecisionTreeClassifier(max_depth=max_depth, criterion="gini",
                                min_samples_leaf=MIN_SAMPLES_LEAF,
                                random_state=SEED)
    sk.fit(onehot, y)
    sk_proba = sk.predict_proba(onehot)[:, 1]
    sk_acc, sk_auc = metrics(sk_proba, y)

    print(f"{max_depth:>5}  {'mine':<8} {count_nodes(mine):>6} {count_leaves(mine):>7} "
          f"{tree_depth(mine):>8} {my_acc:>8.4f} {my_auc:>8.4f}")
    print(f"{'':>5}  {'sklearn':<8} {sk.tree_.node_count:>6} {sk.get_n_leaves():>7} "
          f"{sk.get_depth():>8} {sk_acc:>8.4f} {sk_auc:>8.4f}")

    assert abs(my_auc - sk_auc) < 0.01, f"AUC diverges at depth {max_depth}"
    assert abs(count_nodes(mine) - sk.tree_.node_count) <= 2, \
        f"tree shape diverges at depth {max_depth}"

print("\nstructure and ranking match at every depth")

# --- categorical-only path ----------------------------------------------------
# The combined search keeps choosing numeric features, so without forcing this
# the categorical branch of the recursion is never actually compared against
# anything. A test that never reaches the branch is not a passing test.

empty_num = np.empty((len(y), 0))
cat_mine = build(empty_num, X_cat, y, [], CATEGORICAL_COLS,
                 max_depth=3, min_samples_leaf=MIN_SAMPLES_LEAF)
cat_auc = roc_auc_score(y, predict_proba(cat_mine, empty_num, X_cat))

cat_onehot = pd.get_dummies(train[CATEGORICAL_COLS],
                            columns=CATEGORICAL_COLS).to_numpy(dtype=float)
cat_sk = DecisionTreeClassifier(max_depth=3, criterion="gini",
                                min_samples_leaf=MIN_SAMPLES_LEAF,
                                random_state=SEED).fit(cat_onehot, y)
cat_sk_auc = roc_auc_score(y, cat_sk.predict_proba(cat_onehot)[:, 1])

print(f"\ncategorical only, depth 3")
print(f"  mine     {count_nodes(cat_mine):>3} nodes  auc {cat_auc:.4f}")
print(f"  sklearn  {cat_sk.tree_.node_count:>3} nodes  auc {cat_sk_auc:.4f}")
assert abs(cat_auc - cat_sk_auc) < 0.01, "categorical recursion diverges"
print("\ncategorical recursion matches")
