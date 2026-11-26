"""Figures for the journal version (paper/journal/figures).

Run from the repo root after the experiments (docs/results/*.md):

    python paper/journal/make_journal_figures.py

Figures 1-4 reuse paper/make_figures.py with the 10-match outputs (figure 1 is copied from paper/figures, it needs
network access to rebuild). Figure 5 is new: the correction on a second league's cameras and the leak tolerance of
the visible-player count.

Needs: outputs/stadiums/{idsse,metrica}/stadium_per_camera.csv, outputs/correction/correction_per_broadcast.csv,
outputs/b1_visible/counts_tracks.parquet, outputs/cross_league/cross_league_per_camera.csv,
outputs/s5_replays/per_game.csv.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_figures as mf
import numpy as np
import polars as pl

plt = mf.plt
OUT = Path("paper/journal/figures")


def figure_generalisation() -> None:
    """Fig. 5: (a) Swiss (SN-GSR) cameras against the A-League correction line; (b) leak tolerance."""
    from pitchlens.analytics.broadcast_shots import leak_shift

    t = pl.read_csv("outputs/correction/correction_per_broadcast.csv")
    al = t.group_by("camera").agg(pl.col("players_visible").mean(), pl.col("bias").mean())
    ch = pl.read_csv("outputs/cross_league/cross_league_per_camera.csv")
    k, b = np.polyfit(t["players_visible"], t["bias"], 1)
    lo, hi = float(t["players_visible"].min()), float(t["players_visible"].max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax1.axvspan(lo, hi, color=mf.GRID, alpha=0.6, zorder=0, lw=0)
    ax1.text((lo + hi) / 2, -8, "fitted range", ha="center", fontsize=6.5, color=mf.INK2)
    ax1.scatter(al["players_visible"], al["bias"], s=18, marker="o", facecolor=mf.BLUE, edgecolor="white",
                linewidths=0.6, zorder=3, label=f"A-League cameras ({al.height})")
    ax1.scatter(ch["players_visible"], ch["bias"], s=22, marker="s", facecolor=mf.ORANGE, edgecolor="white",
                linewidths=0.6, zorder=3, label=f"Swiss cameras ({ch.height})")
    xs = np.linspace(9.3, 17.2, 2)
    ax1.plot(xs, k * xs + b, color=mf.INK2, lw=1.0, ls="--", zorder=2, label=f"A-League line ({k:.1f} pp/player)")
    ax1.set_xlabel("Players visible per frame (camera mean)")
    ax1.set_ylabel("Bias in active pressers (%)")
    ax1.set_ylim(top=-5)
    ax1.legend(loc="lower right", fontsize=6.5)
    ax1.set_title("(a) A second league's cameras", fontsize=8)

    pg = pl.read_csv("outputs/s5_replays/per_game.csv")
    share = float(1 - pg["main_live"].median())
    leaks = np.linspace(0, 0.1, 101)
    for v, ls in [(0.0, "-"), (3.0, "--"), (8.0, ":")]:
        ax2.plot(leaks * 100, [leak_shift(share, lk, 14.0, v) for lk in leaks], color=mf.BLUE, ls=ls, lw=1.3,
                 label=f"leaked frames show {v:.0f} players")
    ax2.axhspan(-1, 1, color=mf.GRID, alpha=0.8, zorder=0, lw=0)
    ax2.text(9.8, 0.2, "±1 pp", ha="right", fontsize=6.5, color=mf.INK2)
    ax2.set_xlabel("Non-main-camera frames kept by the filter (%)")
    ax2.set_ylabel("Shift of corrected value (pp)")
    ax2.legend(loc="lower left", fontsize=6.5)
    ax2.set_title(f"(b) Leak tolerance (typical game, {share:.0%} non-main)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_generalisation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    mf.OUT = OUT
    per_camera = mf._per_camera()
    mf.figure_visibility(per_camera)
    mf.figure_lag(per_camera)
    mf.figure_real_video()
    shutil.copy("paper/figures/fig1_cameras.pdf", OUT / "fig1_cameras.pdf")
    figure_generalisation()
    print("Saved figures to", OUT)
