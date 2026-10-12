"""Plot aggregate depth-curve and causal-patching results from local run summaries."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/jev-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
COLORS = {
    "JevK5 v0.3": "#176b87",
    "Base Qwen3.5-4B": "#d17a22",
    "decision": "#7b61a8",
    "query_names": "#147d92",
    "bridge_entities": "#bf6b3a",
    "path_premises": "#4d7c56",
    "distractor_premises": "#8b929a",
}
LABELS = {
    "decision": "Decision position",
    "query_names": "Question-name positions",
    "bridge_entities": "Bridge-entity mentions",
    "path_premises": "Path-premise ends",
    "distractor_premises": "Distractor-premise ends",
}


def read_curve(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#26364a",
        "text.color": "#15283c",
        "xtick.color": "#43556a",
        "ytick.color": "#43556a",
        "axes.edgecolor": "#a5b0bb",
        "savefig.facecolor": "#fafbfc",
    })


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=190, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_behavior():
    series = {
        "JevK5 v0.3": read_curve(ROOT / "runs/e1-jevk5-v03/behavior.csv"),
        "Base Qwen3.5-4B": read_curve(ROOT / "runs/e1-qwen35-4b/behavior.csv"),
    }
    fig, ax = plt.subplots(figsize=(9.2, 5.8), facecolor="#fafbfc")
    ax.set_facecolor("#fafbfc")
    offsets = {"JevK5 v0.3": -.045, "Base Qwen3.5-4B": .045}
    for name, rows in series.items():
        x = np.array([int(r["d"]) for r in rows], dtype=float) + offsets[name]
        y = np.array([float(r["acc"]) * 100 for r in rows])
        lo = np.array([float(r["acc_lo"]) * 100 for r in rows])
        hi = np.array([float(r["acc_hi"]) * 100 for r in rows])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], color=COLORS[name], marker="o",
                    lw=2.4, markersize=7, capsize=4, elinewidth=1.3, label=name)
    ax.axhline(50, color="#9aa6b2", ls="--", lw=1.2, label="50% reference")
    ax.set(xlim=(.65, 6.35), ylim=(30, 85), xticks=range(1, 7),
           xlabel="Shortest chain length (hops)", ylabel="Accuracy (%)")
    ax.grid(axis="y", alpha=.18)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.set_title("Accuracy falls toward chance beyond two hops", loc="left", fontsize=17,
                 fontweight="bold", pad=17)
    ax.text(0, 1.02, "300 balanced test questions per depth and model; error bars are pointwise 95% bootstrap intervals.",
            transform=ax.transAxes, fontsize=9.5, color="#52667c")
    fig.tight_layout()
    save(fig, "03-depth-performance")


def plot_patch_effects():
    data = json.loads((ROOT / "runs/composition-patching/summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True, facecolor="#fafbfc")
    for ax, layer in zip(axes, (16, 20)):
        ax.set_facecolor("#fafbfc")
        for group in LABELS:
            rows = sorted((r for r in data["effects"] if r["layer"] == layer and r["group"] == group),
                          key=lambda r: r["d"])
            if not rows:
                continue
            x = np.array([r["d"] for r in rows])
            y = np.array([r["mean_signed_logit_change"] for r in rows])
            ci = np.array([r["mean_signed_change_ci95"] for r in rows]).T
            ax.errorbar(x, y, yerr=[y - ci[0], ci[1] - y], color=COLORS[group], marker="o",
                        lw=1.9, markersize=5.5, capsize=3, elinewidth=1.1,
                        label=LABELS[group])
        ax.axhline(0, color="#9aa6b2", ls="--", lw=1.0)
        ax.set(title=f"Layer {layer}", xlim=(.7, 6.3), xticks=range(1, 7),
               xlabel="Source chain length (hops)")
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Answer-signed logit change toward source answer")
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Patching effects depend on the layer and token position", x=.07, ha="left",
                 fontsize=17, fontweight="bold", y=1.04)
    fig.text(.07, .965,
             "Source activations patched into an aligned opposite-answer prompt; 200 pairs per source depth. Bars: pointwise 95% stratified bootstrap intervals.",
             fontsize=9.5, color="#52667c")
    fig.text(.07, .015,
             "Bridge mentions are absent at d=1. Intervals are pointwise; no multiple-comparison adjustment.",
             fontsize=9, color="#52667c")
    fig.tight_layout(rect=(0, .05, 1, .91))
    save(fig, "04-patch-effects")


def main():
    style()
    plot_behavior()
    plot_patch_effects()
    print(OUT)


if __name__ == "__main__":
    main()
