"""Figures and tables for the report. Reads saved results only; fits nothing."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#b8b7b2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": MUTED,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})

tuning_grid = pd.read_csv("results/tuning.csv")
benchmark = pd.read_csv("results/benchmark.csv")
with open("results/tuning.json") as fh:
    tuning = json.load(fh)
cfg = tuning["selected"]

y_test = np.load("results/test_y.npy")
proba_mine = np.load("results/test_proba_mine.npy")
proba_sk = np.load("results/test_proba_sklearn.npy")
BASE_RATE = y_test.mean()


# --- Figure 1: confusion matrix ----------------------------------------------
# shade by row rate; by count the TN cell would wash out the rest
def confusion_panel(ax, row, title):
    cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
    rates = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{100 * rates[i, j]:.1f}%",
                    ha="center", va="center", fontsize=11,
                    color="#ffffff" if rates[i, j] > 0.5 else INK)
    ax.set_xticks([0, 1], ["predicted\nnot readmitted", "predicted\n<30 days"])
    ax.set_yticks([0, 1], ["actually\nnot readmitted", "actually\n<30 days"])
    ax.set_title(title, color=INK, fontsize=11, pad=10)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


sel = benchmark[benchmark["selected"]]
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, model in zip(axes, ("mine", "sklearn")):
    row = sel[sel["model"] == model].iloc[0]
    confusion_panel(ax, row, f"{'from-scratch tree' if model == 'mine' else 'sklearn'}"
                             f"\nF1 {row['f1']:.3f} · recall {row['recall']:.3f}")
fig.suptitle(f"Confusion matrices on the held-out test set (n={len(y_test):,}, "
             f"threshold {cfg['threshold']:.3f})", color=INK, fontsize=12)
fig.tight_layout()
fig.savefig("figures/confusion_matrices.png", dpi=200)
plt.close(fig)


# --- Figure 2: depth vs F1, train against validation ------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
grid = tuning_grid[tuning_grid["age_numeric"] == cfg["age_numeric"]]
for col, colour, label in (("train_f1", ORANGE, "train"),
                           ("f1", BLUE, "validation")):
    g = grid.groupby("max_depth")[col].max()
    ax.plot(g.index, g.values, color=colour, linewidth=2, marker="o",
            markersize=5, label=label)
    ax.annotate(label, (g.index[-1], g.values[-1]), xytext=(9, 0),
                textcoords="offset points", color=colour, fontsize=10, va="center")

best_depth = cfg["max_depth"]
ax.axvline(best_depth, color=MUTED, linewidth=1, linestyle="--", zorder=0)
ax.annotate(f"selected depth {best_depth}", (best_depth, ax.get_ylim()[0]),
            xytext=(4, 8), textcoords="offset points", color=INK_2, fontsize=9)
ax.set_xlabel("maximum depth")
ax.set_ylabel("F1 (positive class)")
ax.set_title("Depth vs. F1: the gap between the curves is the overfitting",
             color=INK, fontsize=12, pad=10)
ax.set_xlim(1, tuning_grid["max_depth"].max() + 3)
fig.tight_layout()
fig.savefig("figures/depth_vs_f1.png", dpi=200)
plt.close(fig)


# --- Figure 3: threshold vs precision / recall / F1 ---------------------------
from tuning import threshold_sweep  # noqa: E402

sweep = threshold_sweep(y_test, proba_mine).sort_values("threshold")
fig, ax = plt.subplots(figsize=(7, 4.2))
# stagger labels where the lines meet
for col, colour, dy, label in (("precision", BLUE, 0, "precision"),
                               ("f1", AQUA, 9, "F1"),
                               ("recall", ORANGE, -9, "recall")):
    ax.plot(sweep["threshold"], sweep[col], color=colour, linewidth=2,
            label=label, drawstyle="steps-post")
    ax.annotate(label, (sweep["threshold"].iloc[-1], sweep[col].iloc[-1]),
                xytext=(7, dy), textcoords="offset points", color=colour,
                fontsize=10, va="center")

ax.axvline(cfg["threshold"], color=INK_2, linewidth=1.2, linestyle="--", zorder=0)
ax.annotate(f"tuned threshold {cfg['threshold']:.3f}\n(chosen on validation)",
            (cfg["threshold"], 0.95), xytext=(8, 0), textcoords="offset points",
            color=INK_2, fontsize=9, va="top")
ax.axvline(0.5, color=INK_2, linewidth=1, linestyle=":", zorder=0)
ax.annotate("default 0.5:\nno positives predicted", (0.5, 0.62), xytext=(-8, 0),
            textcoords="offset points", color=INK_2, fontsize=9, va="center",
            ha="right")
ax.set_xlabel("decision threshold")
ax.set_ylabel("score")
ax.set_xlim(0, 0.54)
ax.set_ylim(0, 1.02)
ax.set_title("Why the threshold has to move: scores across the operating range",
             color=INK, fontsize=12, pad=10)
fig.tight_layout()
fig.savefig("figures/threshold_sweep.png", dpi=200)
plt.close(fig)


# --- Figure 4: precision-recall curve -----------------------------------------
# sklearn thick underneath, mine thin on top, to show they coincide
fig, ax = plt.subplots(figsize=(7, 4.2))
for proba, colour, width, label in ((proba_sk, ORANGE, 5, "sklearn"),
                                    (proba_mine, BLUE, 2, "from-scratch tree")):
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ax.plot(recall, precision, color=colour, linewidth=width, label=label,
            solid_capstyle="round")

ax.axhline(BASE_RATE, color=MUTED, linewidth=1.2, linestyle="--", zorder=0)
ax.annotate(f"no-skill baseline ({100 * BASE_RATE:.2f}% positive)",
            (0.02, BASE_RATE), xytext=(0, 6), textcoords="offset points",
            color=INK_2, fontsize=9)
ax.legend(loc="upper right", labelcolor=[ORANGE, BLUE])
ax.set_xlabel("recall")
ax.set_ylabel("precision")
ax.set_xlim(0, 1)
ax.set_ylim(0, max(0.45, BASE_RATE * 4))
ax.set_title("Precision-recall on test: the two implementations coincide",
             color=INK, fontsize=12, pad=10)
fig.tight_layout()
fig.savefig("figures/precision_recall.png", dpi=200)
plt.close(fig)


# --- Tables -------------------------------------------------------------------
print("Table 2: held-out test set, both models at the tuned threshold\n")
cols = ["age", "model", "nodes", "depth", "accuracy", "precision", "recall",
        "f1", "auc", "fit_seconds"]
print(benchmark[cols].to_string(index=False,
      formatters={c: "{:.4f}".format for c in
                  ["accuracy", "precision", "recall", "f1", "auc", "fit_seconds"]}))

print(f"\n  majority-class baseline: accuracy {100 * (1 - BASE_RATE):.2f}%, "
      f"precision 0.0000, recall 0.0000, F1 0.0000")

print("\n\nTable 3: age encoding, validation and test\n")
for name in ("nominal", "numeric"):
    v = tuning[name]
    t = benchmark[(benchmark["age"] == name) & (benchmark["model"] == "mine")].iloc[0]
    print(f"  age {name:<8} validation F1 {v['val_f1']:.4f}   "
          f"test F1 {t['f1']:.4f}  precision {t['precision']:.4f}  recall {t['recall']:.4f}")
print(f"\n  difference in validation F1: "
      f"{abs(tuning['numeric']['val_f1'] - tuning['nominal']['val_f1']):.4f} "
      f"(within noise, not a result)")

print("\nwrote figures/confusion_matrices.png, depth_vs_f1.png, "
      "threshold_sweep.png, precision_recall.png")


# --- Figure 5: feature importances -------------------------------------------
imp_all = pd.read_csv("results/importances.csv")
# computed from this run so the title stays accurate
agreement = float((imp_all["mine"] - imp_all["sklearn"]).abs().max())
imp = imp_all.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.barh(imp["feature"], imp["mine"], color=BLUE, height=0.62)
for name, value in zip(imp["feature"], imp["mine"]):
    ax.annotate(f"{value:.3f}", (value, name), xytext=(5, 0),
                textcoords="offset points", va="center", color=INK_2, fontsize=9)
ax.set_xlabel("share of total impurity decrease")
ax.set_xlim(0, imp["mine"].max() * 1.18)
ax.grid(axis="y", visible=False)
ax.set_title("Which attributes the tree actually uses\n"
             f"(identical for both implementations, agreeing to {agreement:.1e})",
             color=INK, fontsize=12, pad=10)
fig.tight_layout()
fig.savefig("figures/feature_importances.png", dpi=200)
plt.close(fig)


# --- Figure 6: the top of the tree -------------------------------------------
import pickle  # noqa: E402
import textwrap  # noqa: E402

with open("results/tree.pkl", "rb") as fh:
    tree = pickle.load(fh)

RENDER_DEPTH = 3


def layout(node, depth, positions, counter):
    """Leaves get consecutive x slots; each parent sits between its children."""
    if node.is_leaf or depth >= RENDER_DEPTH:
        x = counter[0]
        counter[0] += 1
    else:
        x = (layout(node.left, depth + 1, positions, counter)
             + layout(node.right, depth + 1, positions, counter)) / 2
    positions[id(node)] = (x, depth)
    return x


def node_label(node):
    n = int(node.counts.sum())
    rate = node.counts[1] / n
    if node.is_leaf:
        head = "leaf"
    elif node.split.kind == "numeric":
        head = f"{node.split.name}\n<= {node.split.value:g}"
    else:
        head = f"{node.split.name}\n== {node.split.value}"
    head = "\n".join(textwrap.wrap(head, 22, break_long_words=False))
    return f"{head}\nn={n:,} · {100 * rate:.1f}%", rate


positions, counter = {}, [0]
layout(tree, 0, positions, counter)

fig, ax = plt.subplots(figsize=(14, 7))
cmap = plt.get_cmap("Blues")


def draw(node, depth):
    x, _ = positions[id(node)]
    label, rate = node_label(node)
    shade = min(rate / 0.35, 1.0)
    ax.text(x, -depth, label, ha="center", va="center", fontsize=7.5,
            color="#ffffff" if shade > 0.6 else INK,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=cmap(0.12 + 0.75 * shade),
                      edgecolor="none"))
    if node.is_leaf or depth >= RENDER_DEPTH:
        return
    for child, branch in ((node.left, "true"), (node.right, "false")):
        cx, _ = positions[id(child)]
        ax.plot([x, cx], [-depth - 0.28, -depth - 0.72], color=MUTED,
                linewidth=1, zorder=0)
        ax.annotate(branch, ((x + cx) / 2, -depth - 0.5), fontsize=7,
                    color=INK_2, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE,
                              edgecolor="none"))
        draw(child, depth + 1)


draw(tree, 0)
ax.set_xlim(-0.7, counter[0] - 0.3)
ax.set_ylim(-RENDER_DEPTH - 0.6, 0.6)
ax.axis("off")
ax.set_title(f"Top {RENDER_DEPTH} levels of the tuned tree "
             f"(depth {cfg['max_depth']}, {int(sel[sel.model == 'mine'].iloc[0]['nodes'])} nodes total)\n"
             "shading is the readmission rate among the rows reaching each node",
             color=INK, fontsize=12, pad=14)
fig.tight_layout()
fig.savefig("figures/tree_top_levels.png", dpi=200)
plt.close(fig)

print("\n\nTable 4: feature importances (top 12)\n")
print(pd.read_csv("results/importances.csv").head(12).to_string(
    index=False, formatters={"mine": "{:.4f}".format, "sklearn": "{:.4f}".format}))

print("\n\nTop 3 levels of the tuned tree\n")
print(open("results/tree.txt").read())

print("wrote figures/feature_importances.png, tree_top_levels.png")
