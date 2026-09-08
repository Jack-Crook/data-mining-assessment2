"""Verify the split search against sklearn depth-1 stumps, numeric then combined."""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from decision_tree import NUMERIC_COLS, best_numeric_feature, best_split, gini

SEED = 42
SUBSAMPLE = 5000

train = pd.read_csv("results/train.csv").sample(SUBSAMPLE, random_state=SEED)
X = train[NUMERIC_COLS].to_numpy(dtype=float)
y = train["readmitted_binary"].to_numpy(dtype=int)

print(f"subsample {len(y):,} rows, {y.sum():,} positive ({100 * y.mean():.2f}%)")
print(f"root gini {gini(y):.6f}\n")

(j, name, threshold), gain = best_numeric_feature(X, y, NUMERIC_COLS)
print(f"mine     {name} <= {threshold:.4f}   gain {gain:.6f}")

stump = DecisionTreeClassifier(max_depth=1, criterion="gini", random_state=SEED)
stump.fit(X, y)
sk_feature = NUMERIC_COLS[stump.tree_.feature[0]]
sk_threshold = stump.tree_.threshold[0]
# sklearn reports weighted impurity decrease scaled by the node's share of the tree
sk_gain = stump.tree_.impurity[0] - (
    stump.tree_.weighted_n_node_samples[1] * stump.tree_.impurity[1]
    + stump.tree_.weighted_n_node_samples[2] * stump.tree_.impurity[2]
) / stump.tree_.weighted_n_node_samples[0]
print(f"sklearn  {sk_feature} <= {sk_threshold:.4f}   gain {sk_gain:.6f}")

assert name == sk_feature, "different feature chosen"
assert abs(gain - sk_gain) < 1e-9, "different gain"
print("\nmatch")

# --- categorical + combined search --------------------------------------------

CATEGORICAL_COLS = [c for c in train.columns
                    if c not in NUMERIC_COLS + ["readmitted_binary"]]
X_cat = train[CATEGORICAL_COLS].astype(str).to_numpy()

split = best_split(X, X_cat, y, NUMERIC_COLS, CATEGORICAL_COLS)
print(f"\nmine     {split.kind:<12} {split.name} = {split.value}   gain {split.gain:.6f}")

# one-hot encoding a categorical and splitting a dummy at <= 0.5 IS one-vs-rest,
# so the comparison stays exact across both feature kinds
onehot = pd.get_dummies(train[NUMERIC_COLS + CATEGORICAL_COLS],
                        columns=CATEGORICAL_COLS)
stump = DecisionTreeClassifier(max_depth=1, criterion="gini", random_state=SEED)
stump.fit(onehot.to_numpy(dtype=float), y)
sk_col = onehot.columns[stump.tree_.feature[0]]
sk_gain = stump.tree_.impurity[0] - (
    stump.tree_.weighted_n_node_samples[1] * stump.tree_.impurity[1]
    + stump.tree_.weighted_n_node_samples[2] * stump.tree_.impurity[2]
) / stump.tree_.weighted_n_node_samples[0]
print(f"sklearn  {sk_col}   gain {sk_gain:.6f}")

assert abs(split.gain - sk_gain) < 1e-9, "different gain across combined search"
print("\ncombined search matches")


# --- categorical only ---------------------------------------------------------
# the combined search above happened to pick a numeric feature, so force a
# categorical-only search to verify that path against sklearn independently

empty_num = np.empty((len(y), 0))
cat_split = best_split(empty_num, X_cat, y, [], CATEGORICAL_COLS)
print(f"\nmine     categorical  {cat_split.name} = {cat_split.value}   "
      f"gain {cat_split.gain:.6f}")

cat_onehot = pd.get_dummies(train[CATEGORICAL_COLS], columns=CATEGORICAL_COLS)
cat_stump = DecisionTreeClassifier(max_depth=1, criterion="gini", random_state=SEED)
cat_stump.fit(cat_onehot.to_numpy(dtype=float), y)
cat_sk_gain = cat_stump.tree_.impurity[0] - (
    cat_stump.tree_.weighted_n_node_samples[1] * cat_stump.tree_.impurity[1]
    + cat_stump.tree_.weighted_n_node_samples[2] * cat_stump.tree_.impurity[2]
) / cat_stump.tree_.weighted_n_node_samples[0]
print(f"sklearn  {cat_onehot.columns[cat_stump.tree_.feature[0]]}   "
      f"gain {cat_sk_gain:.6f}")

assert abs(cat_split.gain - cat_sk_gain) < 1e-9, "different gain on categoricals"
print("\ncategorical search matches")
