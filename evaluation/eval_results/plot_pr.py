#!/usr/bin/env python3
"""Plot canonical-space PR curves (one subplot per class) for all models."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
MODELS = [
    ("baseline", "sweep_baseline.csv", "#1f77b4"),
    ("tam (TA merged)", "sweep_tam.csv", "#2ca02c"),
    ("tam2col (TA merged, 2-col kept)", "sweep_tam2col.csv", "#9467bd"),
    ("3cls (h+f merged)", "sweep_3cls.csv", "#ff7f0e"),
    ("3cls_tam (both)", "sweep_3cls_tam.csv", "#d62728"),
]
CLASSES = ["header-footer", "text-area", "footnote"]


def load(p):
    rows = {c: [] for c in CLASSES}
    with open(p) as f:
        for r in csv.DictReader(f):
            rows[r["class"]].append(
                (float(r["conf"]), float(r["P"]), float(r["R"]), float(r["F1"])))
    return rows


data = {name: load(HERE / fn) for name, fn, _ in MODELS}

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
for ax, cls in zip(axes, CLASSES):
    for name, _, color in MODELS:
        pts = sorted(data[name][cls], key=lambda x: x[2])  # by recall
        R = [p[2] for p in pts]
        P = [p[1] for p in pts]
        ax.plot(R, P, "-", color=color, lw=1.8, label=name)
        bf = max(data[name][cls], key=lambda x: x[3])  # max-F1 point
        ax.plot(bf[2], bf[1], "o", color=color, ms=7,
                markeredgecolor="k", markeredgewidth=0.6)
    ax.set_title(f"{cls}  (canonical, IoU 0.5)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.5, 1.005)
    ax.set_ylim(0.5, 1.005)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8)
fig.suptitle("Canonical 3-class PR curves — dots mark each model's max-F1 point",
             fontsize=12)
fig.tight_layout()
out = HERE / "pr_canonical.png"
fig.savefig(out, dpi=130)
print(f"saved {out}")
