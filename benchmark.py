"""Final evaluation: this implementation against sklearn's, on the held-out test
set, at matched hyperparameters and a matched operating point.

This is the only file that reads results/test.csv. Every hyperparameter and the
decision threshold come from results/tuning.json, chosen on a validation slice
of train in train_tree.py.

Both models are scored at the *same* tuned threshold. Comparing a tuned
threshold against sklearn's default 0.5 would credit this implementation for
the threshold tuning rather than for the tree, and at a 9.10% positive rate
sklearn at 0.5 predicts almost no positives at all -- the comparison would be
meaningless in this implementation's favour.
"""

import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from decision_tree import (build, count_leaves, count_nodes, feature_importances,
                            predict_proba, render_tree, tree_depth)
from tuning import TARGET, confusion, feature_columns, matrices, prf

sys.setrecursionlimit(20000)
SEED = 42

with open("results/tuning.json") as fh:
    tuning = json.load(fh)

results = []

for name in ("nominal", "numeric"):
    cfg = tuning[name]
    selected = cfg == tuning["selected"]
    numeric, categorical = feature_columns(pd.read_csv("results/train.csv", nrows=1),
                                           cfg["age_numeric"])

    train = pd.read_csv("results/train.csv")
    test = pd.read_csv("results/test.csv")
    Xn_tr, Xc_tr, y_tr = matrices(train, numeric, categorical)
    Xn_te, Xc_te, y_te = matrices(test, numeric, categorical)

    print(f"\n=== age {name}{'  (selected)' if selected else ''} ===")
    print(f"depth {cfg['max_depth']}, min_samples_leaf {cfg['min_samples_leaf']}, "
          f"threshold {cfg['threshold']:.4f}")

    # --- this implementation ---------------------------------------------
    t0 = time.perf_counter()
    tree = build(Xn_tr, Xc_tr, y_tr, numeric, categorical,
                 max_depth=cfg["max_depth"], min_samples_leaf=cfg["min_samples_leaf"])
    mine_fit = time.perf_counter() - t0
    mine_proba = predict_proba(tree, Xn_te, Xc_te)

    # --- sklearn, same data, categoricals one-hot encoded -----------------
    # test is reindexed onto train's dummy columns so an unseen level becomes
    # all-zeros rather than silently shifting every column to its right
    tr_oh = pd.get_dummies(train[numeric + categorical], columns=categorical)
    te_oh = pd.get_dummies(test[numeric + categorical], columns=categorical)
    if "age" in numeric:
        from tuning import AGE_MIDPOINT
        tr_oh["age"] = train["age"].map(AGE_MIDPOINT)
        te_oh["age"] = test["age"].map(AGE_MIDPOINT)
    te_oh = te_oh.reindex(columns=tr_oh.columns, fill_value=0)

    sk = DecisionTreeClassifier(criterion="gini", max_depth=cfg["max_depth"],
                                min_samples_leaf=cfg["min_samples_leaf"],
                                random_state=SEED)
    t0 = time.perf_counter()
    sk.fit(tr_oh.to_numpy(dtype=float), y_tr)
    sk_fit = time.perf_counter() - t0
    sk_proba = sk.predict_proba(te_oh.to_numpy(dtype=float))[:, 1]

    for model, proba, fit_s, nodes, leaves, depth in (
            ("mine", mine_proba, mine_fit, count_nodes(tree), count_leaves(tree), tree_depth(tree)),
            ("sklearn", sk_proba, sk_fit, sk.tree_.node_count, sk.get_n_leaves(), sk.get_depth())):
        pred = (proba >= cfg["threshold"]).astype(int)
        precision, recall, f1 = prf(y_te, pred)
        tn, fp, fn, tp = confusion(y_te, pred)
        row = {"age": name, "selected": selected, "model": model,
               "nodes": nodes, "leaves": leaves, "depth": depth,
               "threshold": cfg["threshold"], "accuracy": float((pred == y_te).mean()),
               "precision": precision, "recall": recall, "f1": f1,
               "auc": roc_auc_score(y_te, proba),
               "tn": tn, "fp": fp, "fn": fn, "tp": tp, "fit_seconds": fit_s}
        results.append(row)
        print(f"  {model:<8} nodes {nodes:>4}  acc {row['accuracy']:.4f}  "
              f"prec {precision:.4f}  recall {recall:.4f}  F1 {f1:.4f}  "
              f"AUC {row['auc']:.4f}  fit {fit_s:.1f}s")

    if selected:
        np.save("results/test_proba_mine.npy", mine_proba)
        np.save("results/test_proba_sklearn.npy", sk_proba)
        np.save("results/test_y.npy", y_te)

        # importances are compared per original attribute, so sklearn's one-hot
        # columns are summed back to the column they were expanded from
        mine_imp = feature_importances(tree)
        sk_imp = {}
        for dummy, value in zip(tr_oh.columns, sk.feature_importances_):
            if value > 0:
                col = max((c for c in numeric + categorical
                           if dummy == c or dummy.startswith(c + "_")), key=len)
                sk_imp[col] = sk_imp.get(col, 0.0) + float(value)

        imp = pd.DataFrame([{"feature": f, "mine": v, "sklearn": sk_imp.get(f, 0.0)}
                            for f, v in mine_imp.items()])
        imp.to_csv("results/importances.csv", index=False)
        worst = float((imp["mine"] - imp["sklearn"]).abs().max())
        print(f"  feature importances agree to {worst:.1e}")
        assert worst < 1e-9, "importances diverge on the full model"

        with open("results/tree.txt", "w") as fh:
            fh.write(render_tree(tree, max_depth=3) + "\n")
        print("  wrote results/tree.txt (top 3 levels)")

        import pickle
        with open("results/tree.pkl", "wb") as fh:
            pickle.dump(tree, fh)

df = pd.DataFrame(results)
df.to_csv("results/benchmark.csv", index=False)

# the majority-class baseline the proposal's metrics section is arguing against
y_te = np.load("results/test_y.npy")
print(f"\nmajority-class baseline: accuracy {100 * (1 - y_te.mean()):.2f}%, "
      f"recall 0.0000, F1 0.0000 ({y_te.sum():,} positives never found)")

sel = df[df["selected"]]
assert (sel["f1"] > 0.15).all(), "F1 collapsed on test -- threshold did not transfer"
assert abs(sel[sel.model == "mine"]["auc"].iloc[0]
           - sel[sel.model == "sklearn"]["auc"].iloc[0]) < 0.02, \
    "the two implementations disagree on test -- investigate before reporting"
print("\nwrote results/benchmark.csv")
