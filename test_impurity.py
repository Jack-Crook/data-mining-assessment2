import numpy as np
from decision_tree import gini, weighted_impurity, impurity_decrease

pure = np.array([0, 0, 0, 0])
half = np.array([0, 0, 1, 1])
skew = np.array([0, 0, 0, 1])          # 75/25

assert gini(pure) == 0.0                        # no impurity
assert gini(half) == 0.5                        # maximum for binary
assert abs(gini(skew) - 0.375) < 1e-9           # 1 - (0.75^2 + 0.25^2)

# a perfect split of `half` into two pure children
assert weighted_impurity(np.array([0, 0]), np.array([1, 1])) == 0.0
assert abs(impurity_decrease(half, np.array([0, 0]), np.array([1, 1])) - 0.5) < 1e-9

# a useless split: both children keep the parent's 50/50 mix
assert impurity_decrease(half, np.array([0, 1]), np.array([0, 1])) == 0.0

print("impurity tests pass")