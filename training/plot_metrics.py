"""
plot_metrics.py  ─  Developer training dashboard
─────────────────────────────────────────────────
Run after (or during) training to visualise all logged metrics.

Usage:
    python training/plot_metrics.py
    python training/plot_metrics.py --metrics outputs/metrics.json
    python training/plot_metrics.py --save             # saves PNG instead of showing
"""

import os
import sys
import json
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ── Style ──────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
PALETTE = {
    "G":    "#f97316",   # orange  – Generator
    "D":    "#6366f1",   # indigo  – Discriminator
    "val":  "#22d3ee",   # cyan    – Validation
    "l1":   "#a3e635",   # lime    – L1 component
    "perc": "#e879f9",   # pink    – Perceptual component
    "lr":   "#facc15",   # yellow  – Learning rate
    "time": "#94a3b8",   # slate   – Epoch time
}
BG        = "#0f172a"   # very dark blue background
CARD      = "#1e293b"   # card background
GRID_CLR  = "#334155"
TEXT_CLR  = "#e2e8f0"

matplotlib.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    GRID_CLR,
    "axes.labelcolor":   TEXT_CLR,
    "xtick.color":       TEXT_CLR,
    "ytick.color":       TEXT_CLR,
    "text.color":        TEXT_CLR,
    "grid.color":        GRID_CLR,
    "grid.linewidth":    0.6,
    "legend.framealpha": 0.25,
    "legend.edgecolor":  GRID_CLR,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
})

# ── Helpers ────────────────────────────────────────────────────────────────
def styled_ax(ax, title, xlabel="Epoch", ylabel=""):
    ax.set_title(title, color=TEXT_CLR, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color=TEXT_CLR, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_CLR, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)

def line(ax, x, y, label, color, lw=2, marker="o", ms=4, dashes=None):
    kwargs = dict(label=label, color=color, linewidth=lw,
                  marker=marker, markersize=ms, markerfacecolor=color,
                  markeredgewidth=0)
    if dashes:
        kwargs["dashes"] = dashes
    ax.plot(x, y, **kwargs)
    # Annotate final value
    if len(x) > 0:
        ax.annotate(f"{y[-1]:.4f}", xy=(x[-1], y[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=7, color=color, va="center")

def shade_best(ax, epochs, values, color):
    """Shade the region under the curve lightly."""
    ax.fill_between(epochs, values, alpha=0.08, color=color)

# ── Main ───────────────────────────────────────────────────────────────────
def plot(metrics_path: str, save: bool):
    # ── Load data ──────────────────────────────────────────────────────────
    if not os.path.isfile(metrics_path):
        print(f"[ERROR] Metrics file not found: {metrics_path}")
        print("  Train the model first — metrics are saved automatically.")
        sys.exit(1)

    with open(metrics_path) as f:
        data = json.load(f)

    if len(data) == 0:
        print("[ERROR] metrics.json is empty. Run training first.")
        sys.exit(1)

    # Sort by epoch just in case
    data.sort(key=lambda m: m["epoch"])

    epochs   = [m["epoch"]      for m in data]
    loss_G   = [m["loss_G"]     for m in data]
    loss_D   = [m["loss_D"]     for m in data]
    val_l1   = [m["val_l1"]     for m in data]
    l1_loss  = [m.get("l1_loss",   None) for m in data]
    perc_loss= [m.get("perc_loss", None) for m in data]
    lr_G     = [m.get("lr_G",      None) for m in data]
    lr_D     = [m.get("lr_D",      None) for m in data]
    ep_secs  = [m.get("epoch_secs",None) for m in data]

    total_epochs = epochs[-1]
    best_epoch   = epochs[val_l1.index(min(val_l1))]
    best_val     = min(val_l1)

    # ── Layout ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    fig.suptitle(
        f"GAN Colorizer — Training Metrics Dashboard   "
        f"[{total_epochs} epochs | best val L1 = {best_val:.4f} @ epoch {best_epoch}]",
        fontsize=14, fontweight="bold", color=TEXT_CLR, y=0.98
    )

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.32,
                           left=0.06, right=0.97,
                           top=0.92,  bottom=0.07)

    # ── Plot 1: Generator vs Discriminator Loss ─────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    styled_ax(ax1, "Generator & Discriminator Loss", ylabel="Avg Loss / Epoch")
    line(ax1, epochs, loss_G, "Generator (G)",     PALETTE["G"])
    line(ax1, epochs, loss_D, "Discriminator (D)", PALETTE["D"])
    shade_best(ax1, epochs, loss_G, PALETTE["G"])
    shade_best(ax1, epochs, loss_D, PALETTE["D"])
    ax1.legend(fontsize=8)

    # ── Plot 2: Validation L1 Loss ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    styled_ax(ax2, "Validation L1 Loss", ylabel="Val L1")
    line(ax2, epochs, val_l1, "Val L1 Loss", PALETTE["val"])
    shade_best(ax2, epochs, val_l1, PALETTE["val"])
    # Mark best epoch
    ax2.axvline(best_epoch, color=PALETTE["val"], linestyle=":", linewidth=1.2, alpha=0.7)
    ax2.annotate(f"Best\nEpoch {best_epoch}", xy=(best_epoch, best_val),
                 xytext=(8, 10), textcoords="offset points",
                 fontsize=7.5, color=PALETTE["val"],
                 arrowprops=dict(arrowstyle="->", color=PALETTE["val"], lw=0.8))
    ax2.legend(fontsize=8)

    # ── Plot 3: G Loss components (L1 + Perceptual) ─────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    styled_ax(ax3, "Generator Loss Components", ylabel="Avg Loss / Epoch")
    has_l1   = any(v is not None for v in l1_loss)
    has_perc = any(v is not None for v in perc_loss)
    if has_l1:
        l1_clean = [v if v is not None else float("nan") for v in l1_loss]
        line(ax3, epochs, l1_clean, "L1 (pixel)",  PALETTE["l1"])
        shade_best(ax3, epochs, l1_clean, PALETTE["l1"])
    if has_perc:
        perc_clean = [v if v is not None else float("nan") for v in perc_loss]
        line(ax3, epochs, perc_clean, "Perceptual (VGG)", PALETTE["perc"])
        shade_best(ax3, epochs, perc_clean, PALETTE["perc"])
    if not has_l1 and not has_perc:
        ax3.text(0.5, 0.5, "No component data\n(retrain to populate)",
                 ha="center", va="center", transform=ax3.transAxes,
                 color=GRID_CLR, fontsize=10)
    ax3.legend(fontsize=8)

    # ── Plot 4: G vs Val L1 overlay (overfitting monitor) ──────────────
    ax4 = fig.add_subplot(gs[1, 0])
    styled_ax(ax4, "Overfitting Monitor  (Train G vs Val L1)", ylabel="Loss")
    line(ax4, epochs, loss_G, "Train G Loss", PALETTE["G"])
    line(ax4, epochs, val_l1, "Val L1 Loss",  PALETTE["val"], dashes=(4, 2))
    shade_best(ax4, epochs, loss_G, PALETTE["G"])
    # Gap annotation
    gap = [abs(g - v) for g, v in zip(loss_G, val_l1)]
    ax4_twin = ax4.twinx()
    ax4_twin.plot(epochs, gap, color="#475569", linewidth=1,
                  linestyle=":", label="|Gap|")
    ax4_twin.set_ylabel("|Train − Val|", color="#475569", fontsize=8)
    ax4_twin.tick_params(colors="#475569", labelsize=7)
    ax4.legend(fontsize=8, loc="upper right")

    # ── Plot 5: Learning Rate schedule ─────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    styled_ax(ax5, "Learning Rate Schedule", ylabel="LR")
    has_lr = any(v is not None for v in lr_G)
    if has_lr:
        lr_G_clean = [v if v is not None else float("nan") for v in lr_G]
        lr_D_clean = [v if v is not None else float("nan") for v in lr_D]
        line(ax5, epochs, lr_G_clean, "LR Generator",     PALETTE["lr"])
        line(ax5, epochs, lr_D_clean, "LR Discriminator", PALETTE["D"], dashes=(4, 2))
        ax5.legend(fontsize=8)
    else:
        ax5.text(0.5, 0.5, "No LR data", ha="center", va="center",
                 transform=ax5.transAxes, color=GRID_CLR, fontsize=10)

    # ── Plot 6: Epoch training time ────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    styled_ax(ax6, "Epoch Duration", ylabel="Seconds")
    has_time = any(v is not None for v in ep_secs)
    if has_time:
        et_clean = [v if v is not None else float("nan") for v in ep_secs]
        ax6.bar(epochs, et_clean, color=PALETTE["time"], alpha=0.75, width=0.7)
        avg_t = sum(v for v in et_clean if v == v) / max(1, sum(1 for v in et_clean if v == v))
        ax6.axhline(avg_t, color=PALETTE["lr"], linestyle="--", linewidth=1, label=f"Avg {avg_t:.0f}s")
        ax6.legend(fontsize=8)
    else:
        ax6.text(0.5, 0.5, "No timing data", ha="center", va="center",
                 transform=ax6.transAxes, color=GRID_CLR, fontsize=10)

    # ── Save or show ───────────────────────────────────────────────────
    if save:
        out_dir  = os.path.dirname(metrics_path)
        out_path = os.path.join(out_dir, "training_metrics.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"[OK] Saved: {out_path}")
    else:
        plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GAN training metrics")
    parser.add_argument("--metrics", type=str, default="outputs/metrics.json",
                        help="Path to metrics.json (default: outputs/metrics.json)")
    parser.add_argument("--save", action="store_true",
                        help="Save plot as PNG instead of opening a window")
    args = parser.parse_args()
    plot(args.metrics, args.save)
