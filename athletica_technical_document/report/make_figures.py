"""Figures for the Athletica Intelligentia technical document.

All numbers are the published results from the README (public five minutes,
official scorer). The stoppage figure is a labelled schematic, not match data.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# Reference palette (light mode)
OURS = "#2a78d6"      # categorical slot 1
ACCENT = "#eb6834"    # categorical slot 2
AQUA = "#1baf7a"      # categorical slot 3
BASE = "#9a9994"      # neutral context series
DEAD = "#d9d8d3"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e0"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
    "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


# 1. Pipeline -----------------------------------------------------------------
def pipeline():
    fig, ax = plt.subplots(figsize=(10.4, 2.1))
    ax.set_axis_off(); ax.set_xlim(0, 104); ax.set_ylim(0, 22)
    steps = [
        ("Baseline", "nearest player's\nteam, per frame", BASE),
        ("Step 1", "detect dead balls\nfrom player slowdown", OURS),
        ("Step 2", "fill hidden-ball gaps,\nnever across a dead ball", OURS),
        ("Smooth", "majority vote\nover ±window", BASE),
        ("Force D", "Step 1 frames\nset to dead ball", OURS),
    ]
    w, gap = 18.0, 3.2
    for i, (title, body, colour) in enumerate(steps):
        x = 1 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 3), w, 15, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc="white", ec=colour, lw=1.6))
        ax.add_patch(FancyBboxPatch((x, 14), w, 4, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc=colour, ec=colour, lw=1.6))
        ax.text(x + w / 2, 16, title, ha="center", va="center", color="white", fontsize=9.5, weight="bold")
        ax.text(x + w / 2, 8.5, body, ha="center", va="center", color=INK, fontsize=7.6, linespacing=1.3)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.4, 10.5), (x + w + gap - 0.4, 10.5),
                                         arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.2))
    ax.text(1, 0.6, "Blue: our additions.  Grey: the challenge baseline, used unchanged.",
            fontsize=7.5, color=INK2)
    save(fig, "fig1_pipeline.png")


# 2. Anatomy of a stoppage (schematic) ----------------------------------------
def stoppage():
    t = np.linspace(0, 60, 1200)
    exit_t, restart_t = 18, 41
    speed = 2.3 + 0.25 * np.sin(t / 2.3)
    drop = 1 / (1 + np.exp(-(t - (exit_t + 3)) / 1.2))
    rise = 1 / (1 + np.exp(-(t - (restart_t + 3)) / 1.3))
    speed = speed - 1.25 * drop * (1 - rise) + 0.05 * np.sin(t * 1.7)
    line = np.where(t < exit_t, 13 - 9 * (t / exit_t) ** 3, np.nan)
    line[(t >= exit_t) & (t < exit_t + 1.5)] = -2
    placed = (t >= 30) & (t < restart_t)
    line[placed] = 4
    after = t >= restart_t
    line[after] = 4 + 9 * (1 - np.exp(-(t[after] - restart_t) / 4))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.2, 3.9), sharex=True, gridspec_kw={"hspace": 0.18})
    for ax in (a1, a2):
        ax.axvspan(exit_t, restart_t + 1, color=DEAD, lw=0, zorder=0)
        ax.axvline(exit_t, color=ACCENT, lw=1.2, ls="--")
        ax.axvline(restart_t, color=AQUA, lw=1.2, ls="--")
    a1.plot(t, speed, color=OURS, lw=2)
    a1.axhline(1.5, color=INK2, lw=0.9, ls=":")
    a1.text(0.5, 1.56, "slowdown threshold 1.5 m/s", fontsize=7.5, color=INK2, va="bottom")
    a1.set_ylabel("mean player\nspeed (m/s)"); a1.set_ylim(0.5, 3.7)
    a1.text(exit_t + 0.6, 3.62, "ball exit\n(dead ball starts)", color=ACCENT, fontsize=8, va="top")
    a1.text(restart_t + 0.6, 3.62, "set-piece start + 1 s\n(dead ball ends)", color="#0d7a54", fontsize=8, va="top")
    a1.text((exit_t + restart_t) / 2, 0.62, "D", ha="center", fontsize=13, weight="bold", color=INK2)
    a2.plot(t, line, color=OURS, lw=2)
    a2.axhline(0, color=INK2, lw=0.9)
    a2.text(59.5, 0.4, "pitch line", ha="right", fontsize=7.5, color=INK2, va="bottom")
    a2.text(exit_t + 1.8, -2.6, "ball outside", fontsize=7.5, color=INK2, va="center")
    a2.text(24.8, 1.2, "ball not tracked", ha="center", fontsize=7.5, color=INK2, style="italic")
    a2.text(30.5, 5.0, "ball placed for restart", fontsize=7.5, color=INK2)
    a2.set_ylabel("ball to nearest\nline (m)"); a2.set_ylim(-5, 15)
    a2.set_xlabel("seconds"); a2.set_xlim(0, 60)
    fig.suptitle("Schematic of a typical stoppage (illustrative, not match data)", x=0.12, ha="left",
                 fontsize=9.5, color=INK, y=0.98)
    save(fig, "fig2_stoppage.png")


# 3. Signal strength (AUC) -----------------------------------------------------
def signals():
    names = ["Mean player speed", "Ball distance to nearest line", "Ball stationary time"]
    auc = [0.79, 0.77, 0.51]
    fig, ax = plt.subplots(figsize=(6.4, 1.9))
    y = np.arange(len(names))[::-1]
    colours = [OURS, OURS, BASE]
    ax.barh(y, [a - 0.5 for a in auc], left=0.5, height=0.55, color=colours)
    for yi, a in zip(y, auc):
        ax.text(a + 0.006, yi, "%.2f" % a, va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlim(0.5, 0.85); ax.set_xlabel("separation strength (AUC folded onto 0.5-1; 0.5 = useless)")
    ax.grid(axis="y", visible=False)
    ax.set_title("How well each signal separates dead from live frames", loc="left", fontsize=9.5)
    save(fig, "fig3_signals.png")


# 4. Results across smoothing windows -----------------------------------------
def windows():
    w = ["±12", "±25", "±50", "±75", "±125", "±200"]
    spell_b = [0.101, 0.287, 0.326, 0.400, 0.424, 0.193]
    spell_o = [0.106, 0.298, 0.446, 0.520, 0.486, 0.288]
    macro_b = [0.607, 0.620, 0.621, 0.603, 0.581, 0.515]
    macro_o = [0.768, 0.788, 0.811, 0.818, 0.812, 0.789]
    x = np.arange(len(w))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 2.9))
    for ax, b, o, title, ylim in [(axes[0], spell_b, spell_o, "spellF1 (ranking metric)", (0, 0.6)),
                                  (axes[1], macro_b, macro_o, "macroF1", (0.4, 0.9))]:
        ax.axvspan(1.6, 2.4, color="#f1f0ec", lw=0, zorder=0)
        ax.plot(x, b, color=BASE, lw=2, marker="o", ms=5, mec="white", mew=1.2, label="baseline")
        ax.plot(x, o, color=OURS, lw=2, marker="o", ms=5, mec="white", mew=1.2, label="ours")
        ax.set_xticks(x); ax.set_xticklabels(w); ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", fontsize=9.5); ax.set_xlabel("smoothing window (frames)")
        ax.text(x[-1] + 0.12, o[-1], "ours", color=INK, fontsize=8, va="center")
        ax.text(x[-1] + 0.12, b[-1], "baseline", color=INK2, fontsize=8, va="center")
        ax.set_xlim(-0.3, len(w) - 0.3)
        ax.text(2, ylim[1], "published\nbaseline window", ha="center", va="top", fontsize=7, color=INK2)
    axes[1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    save(fig, "fig4_windows.png")


# 5. What we tried --------------------------------------------------------------
def ablation():
    rows = [  # name, spellF1, macroF1, kind
        ("Step 1 + Step 2 (our solution)", 0.446, 0.811, "ours"),
        ("Step 2 only: gap filling", 0.379, 0.614, "kept"),
        ("Step 1 only: dead-ball detection", 0.369, 0.809, "kept"),
        ("First version of Step 1", 0.344, 0.644, "dropped"),
        ("Baseline", 0.326, 0.621, "baseline"),
        ("Restart on acceleration", 0.322, 0.803, "dropped"),
        ("“Hold owner”", 0.297, 0.598, "dropped"),
    ]
    colour = {"ours": OURS, "kept": "#8fb7e8", "dropped": "#cfcec9", "baseline": BASE}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 2.9), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, k, title, xmax in [(axes[0], 1, "spellF1", 0.55), (axes[1], 2, "macroF1", 0.95)]:
        vals = [r[k] for r in rows]
        ax.barh(y, vals, height=0.62, color=[colour[r[3]] for r in rows])
        for yi, v in zip(y, vals):
            ax.text(v + xmax * 0.01, yi, "%.3f" % v, va="center", fontsize=7.8, color=INK)
        ax.axvline(rows[4][k], color=INK2, lw=0.8, ls=":")
        ax.set_xlim(0, xmax); ax.set_title(title, loc="left", fontsize=9.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y); axes[0].set_yticklabels([r[0] for r in rows])
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=OURS, label="final solution"), Patch(color="#8fb7e8", label="kept step"),
                            Patch(color=BASE, label="baseline"), Patch(color="#cfcec9", label="tried and dropped")],
                   loc="lower center", bbox_to_anchor=(0.6, -0.07), ncol=4, fontsize=7.5)
    fig.suptitle("Public five minutes, ±50 smoothing. Dotted line: baseline.", x=0.0, ha="left",
                 fontsize=8.5, color=INK2, y=1.0)
    fig.tight_layout()
    save(fig, "fig5_ablation.png")


# 6. Possession rules for missing-ball gaps --------------------------------------
def gaps():
    H, A, D, U = OURS, ACCENT, DEAD, "white"
    cases = [
        ("Same team either side,\ngap ≤ 20 s", [H, U, H], [H, H, H], "filled with that team"),
        ("Teams differ,\ngap ≤ 2 s", [H, U, A], [H, (H, A), A], "split down the middle"),
        ("Teams differ,\ngap > 2 s", [H, U, A], [H, D, A], "left as dead ball"),
        ("Gap inside a Step 1\ndead ball", [H, U, H], [H, D, H], "never filled"),
    ]
    fig, ax = plt.subplots(figsize=(10.2, 2.9))
    ax.set_axis_off(); ax.set_xlim(0, 108); ax.set_ylim(0, len(cases) * 10 + 6)
    widths = [6, 10, 6]

    def draw(x0, yb, segs):
        x = x0
        for wdt, s in zip(widths, segs):
            if isinstance(s, tuple):
                ax.add_patch(plt.Rectangle((x, yb), wdt / 2, 5, fc=s[0], ec="white", lw=1.5))
                ax.add_patch(plt.Rectangle((x + wdt / 2, yb), wdt / 2, 5, fc=s[1], ec="white", lw=1.5))
            else:
                ax.add_patch(plt.Rectangle((x, yb), wdt, 5, fc=s, ec=MUTED if s == U else "white",
                                           lw=0.8 if s == U else 1.5, ls="--" if s == U else "-"))
            x += wdt
    ax.text(30, len(cases) * 10 + 3.5, "before", ha="center", fontsize=8, color=INK2)
    ax.text(62, len(cases) * 10 + 3.5, "after", ha="center", fontsize=8, color=INK2)
    for i, (label, before, after, verdict) in enumerate(cases):
        yb = (len(cases) - 1 - i) * 10 + 2
        ax.text(0, yb + 2.5, label, va="center", fontsize=8, color=INK, linespacing=1.2)
        draw(19, yb, before)
        ax.add_patch(FancyArrowPatch((42, yb + 2.5), (49, yb + 2.5), arrowstyle="-|>", mutation_scale=10,
                                     color=MUTED, lw=1))
        draw(51, yb, after)
        ax.text(75, yb + 2.5, verdict, va="center", fontsize=8.5, color=INK)
    for x, c, n in [(75, H, "H: Barcelona"), (91, A, "A: Athletic")]:
        ax.add_patch(plt.Rectangle((x, len(cases) * 10 + 2.6), 1.8, 1.8, fc=c, ec="none"))
        ax.text(x + 2.4, len(cases) * 10 + 3.5, n, va="center", fontsize=7.5, color=INK2)
    ax.add_patch(plt.Rectangle((75, len(cases) * 10 - 0.4), 1.8, 1.8, fc=D, ec="none"))
    ax.text(77.4, len(cases) * 10 + 0.5, "D: dead ball", va="center", fontsize=7.5, color=INK2)
    ax.add_patch(plt.Rectangle((91, len(cases) * 10 - 0.4), 1.8, 1.8, fc=U, ec=MUTED, ls="--", lw=0.8))
    ax.text(93.4, len(cases) * 10 + 0.5, "ball missing", va="center", fontsize=7.5, color=INK2)
    save(fig, "fig6_gaps.png")


if __name__ == "__main__":
    pipeline(); stoppage(); signals(); windows(); ablation(); gaps()
    print(sorted(os.listdir(OUT)))
