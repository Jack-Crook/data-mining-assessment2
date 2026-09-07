"""Figures and tables for the report's results section.

Reads only what the earlier scripts wrote (results/tuning.csv, benchmark.csv and
the saved test probabilities); fits nothing, so the test set is not re-touched.

Palette is the three-slot categorical set validated for all-pairs separation
under deuteranopia and tritanopia. Every series is also direct-labelled, so
identity never rests on colour alone -- which matters for a report that may be
printed in greyscale.
"""

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
# Cells are shaded by row rate, not raw count: at 9% positive the true-negative
# cell is two orders of magnitude larger than the rest, so shading by count
# would render every other cell white and show nothing.
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


# --- Figure 2: depth vs validation F1 ----------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
# the two encodings track each other almost exactly, so the wider line is drawn
# underneath: plotting them at equal weight would simply hide whichever went first
for age_numeric, colour, width, label in ((True, ORANGE, 5, "age numeric"),
                                          (False, BLUE, 2, "age nominal")):
    g = (tuning_grid[tuning_grid["age_numeric"] == age_numeric]
         .groupby("max_depth")["f1"].max())
    ax.plot(g.index, g.values, color=colour, linewidth=width, marker="o",
            markersize=5, label=label, solid_capstyle="round")
    ax.annotate(label, (g.index[-1], g.values[-1]),
                xytext=(9, 7 if age_numeric else -7), textcoords="offset points",
                color=colour, fontsize=10, va="center")

best_depth = cfg["max_depth"]
ax.axvline(best_depth, color=MUTED, linewidth=1, linestyle="--", zorder=0)
ax.annotate(f"selected depth {best_depth}", (best_depth, ax.get_ylim()[0]),
            xytext=(4, 8), textcoords="offset points", color=INK_2, fontsize=9)
ax.set_xlabel("maximum depth")
ax.set_ylabel("best validation F1 (positive class)")
ax.set_title("Depth vs. validation F1 — best over min_samples_leaf at each depth",
             color=INK, fontsize=12, pad=10)
# note the y-axis is zoomed: the gap between the two encodings is ~0.0005, which
# is noise. The shape of the curve is the finding here, not the gap.
ax.set_xlim(1, tuning_grid["max_depth"].max() + 2.5)
fig.tight_layout()
fig.savefig("figures/depth_vs_f1.png", dpi=200)
plt.close(fig)


# --- Figure 3: threshold vs precision / recall / F1 ---------------------------
from tuning import threshold_sweep  # noqa: E402

sweep = threshold_sweep(y_test, proba_mine).sort_values("threshold")
fig, ax = plt.subplots(figsize=(7, 4.2))
# F1 and recall converge at the right-hand end, so the label offsets are
# staggered rather than all centred on their line
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
ax.annotate("default 0.5 —\nno positives predicted", (0.5, 0.62), xytext=(-8, 0),
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
# sklearn is drawn thick underneath and this implementation thin on top: the
# point of the figure is that the two curves coincide exactly.
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
ax.set_title("Precision-recall on test — the two implementations coincide",
             color=INK, fontsize=12, pad=10)
fig.tight_layout()
fig.savefig("figures/precision_recall.png", dpi=200)
plt.close(fig)


# --- Tables -------------------------------------------------------------------
print("Table 2 — held-out test set, both models at the tuned threshold\n")
cols = ["age", "model", "nodes", "depth", "accuracy", "precision", "recall",
        "f1", "auc", "fit_seconds"]
print(benchmark[cols].to_string(index=False,
      formatters={c: "{:.4f}".format for c in
                  ["accuracy", "precision", "recall", "f1", "auc", "fit_seconds"]}))

print(f"\n  majority-class baseline: accuracy {100 * (1 - BASE_RATE):.2f}%, "
      f"precision 0.0000, recall 0.0000, F1 0.0000")

print("\n\nTable 3 — age encoding, validation and test\n")
for name in ("nominal", "numeric"):
    v = tuning[name]
    t = benchmark[(benchmark["age"] == name) & (benchmark["model"] == "mine")].iloc[0]
    print(f"  age {name:<8} validation F1 {v['val_f1']:.4f}   "
          f"test F1 {t['f1']:.4f}  precision {t['precision']:.4f}  recall {t['recall']:.4f}")
print(f"\n  difference in validation F1: "
      f"{abs(tuning['numeric']['val_f1'] - tuning['nominal']['val_f1']):.4f} "
      f"— within noise, not a result")

print("\nwrote figures/confusion_matrices.png, depth_vs_f1.png, "
      "threshold_sweep.png, precision_recall.png")
