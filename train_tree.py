"""Tune depth, min_samples_leaf and the decision threshold on a validation slice
carved out of the training set, for both `age` encodings.

The held-out test set from split.py is not read anywhere in this file. Every
choice made here is made on data the final numbers were not measured on.
Tuning on test is exactly the methodological failure the Summary section would
have to own up to.
"""

import json
import sys
import time

import pandas as pd
from sklearn.model_selection import train_test_split

from decision_tree import build, count_nodes, predict_proba, tree_depth
from tuning import (TARGET, best_threshold, feature_columns, matrices, prf)

sys.setrecursionlimit(20000)

SEED = 42
VAL_SIZE = 0.2
DEPTHS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
MIN_LEAVES = [20, 50, 100, 200]

train_full = pd.read_csv("results/train.csv")
fit_df, val_df = train_test_split(train_full, test_size=VAL_SIZE,
                                  stratify=train_full[TARGET], random_state=SEED)

print(f"fit   {len(fit_df):>6,} rows  {100 * fit_df[TARGET].mean():5.2f}% positive")
print(f"val   {len(val_df):>6,} rows  {100 * val_df[TARGET].mean():5.2f}% positive")
assert abs(fit_df[TARGET].mean() - val_df[TARGET].mean()) < 0.005

rows = []
chosen = {}

for age_numeric in (False, True):
    label = "age numeric (midpoints)" if age_numeric else "age nominal (buckets)"
    numeric, categorical = feature_columns(train_full, age_numeric)
    Xn_fit, Xc_fit, y_fit = matrices(fit_df, numeric, categorical)
    Xn_val, Xc_val, y_val = matrices(val_df, numeric, categorical)

    print(f"\n=== {label} ===")
    print(f"{'depth':>5} {'min_leaf':>9} {'nodes':>7} {'thresh':>8} "
          f"{'prec':>7} {'recall':>7} {'val F1':>7} {'train F1':>8} {'fit s':>7}")

    for max_depth in DEPTHS:
        for min_samples_leaf in MIN_LEAVES:
            t0 = time.perf_counter()
            tree = build(Xn_fit, Xc_fit, y_fit, numeric, categorical,
                         max_depth=max_depth, min_samples_leaf=min_samples_leaf)
            fit_s = time.perf_counter() - t0

            proba = predict_proba(tree, Xn_val, Xc_val)
            threshold, f1 = best_threshold(y_val, proba)
            precision, recall, _ = prf(y_val, (proba >= threshold).astype(int))

            # the same tree scored on the data it was fitted on, at the same
            # threshold. Two curves that separate as depth grows is what
            # overfitting looks like; the validation curve alone only shows
            # where to stop, not why.
            train_proba = predict_proba(tree, Xn_fit, Xc_fit)
            _, _, train_f1 = prf(y_fit, (train_proba >= threshold).astype(int))

            rows.append({"age_numeric": age_numeric, "max_depth": max_depth,
                         "min_samples_leaf": min_samples_leaf,
                         "nodes": count_nodes(tree), "depth_reached": tree_depth(tree),
                         "threshold": threshold, "precision": precision,
                         "recall": recall, "f1": f1, "train_f1": train_f1,
                         "fit_seconds": fit_s})

            print(f"{max_depth:>5} {min_samples_leaf:>9} {count_nodes(tree):>7} "
                  f"{threshold:>8.4f} {precision:>7.4f} {recall:>7.4f} {f1:>7.4f} "
                  f"{train_f1:>8.4f} {fit_s:>7.1f}")

    grid = pd.DataFrame([r for r in rows if r["age_numeric"] == age_numeric])
    best = grid.loc[grid["f1"].idxmax()]
    chosen["numeric" if age_numeric else "nominal"] = {
        "age_numeric": age_numeric,
        "max_depth": int(best["max_depth"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "threshold": float(best["threshold"]),
        "val_f1": float(best["f1"]),
        "val_precision": float(best["precision"]),
        "val_recall": float(best["recall"]),
    }
    print(f"best: depth {int(best['max_depth'])}, min_leaf {int(best['min_samples_leaf'])}, "
          f"threshold {best['threshold']:.4f}, validation F1 {best['f1']:.4f} "
          f"(train F1 {best['train_f1']:.4f})")

pd.DataFrame(rows).to_csv("results/tuning.csv", index=False)

winner = max(chosen.values(), key=lambda c: c["val_f1"])
chosen["selected"] = winner
with open("results/tuning.json", "w") as fh:
    json.dump(chosen, fh, indent=2)

print("\n=== age encoding comparison (validation) ===")
for name, c in (("nominal", chosen["nominal"]), ("numeric", chosen["numeric"])):
    mark = "  <- selected" if c is winner else ""
    print(f"  {name:<8} F1 {c['val_f1']:.4f}  precision {c['val_precision']:.4f}  "
          f"recall {c['val_recall']:.4f}{mark}")

# a threshold left at 0.5 is the failure mode this whole step exists to fix;
# if tuning ever returns 0.5 something has gone wrong upstream
assert winner["threshold"] < 0.5, "tuned threshold should sit well below 0.5"
assert winner["val_f1"] > 0.0, "tuning produced a degenerate all-negative model"
print("\nwrote results/tuning.csv and results/tuning.json")
