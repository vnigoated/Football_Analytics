"""Rebuild the paper figures from saved experiment outputs.

Run from the repo root after the experiments (see docs/results/*.md):

    python paper/make_figures.py

Needs: outputs/stadiums/{idsse,metrica}/stadium_per_camera.csv (pitchlens stadium-summary),
configs/cameras/*.json, and network access for one IDSSE match (figure 1).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

OUT = Path("paper/figures")
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, HIDDEN = "#0b0b0b", "#52514e", "#e4e3df", "#b5b4ad"
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)


def _pitch(ax, L=105.0, W=68.0):
    kw = {"color": INK2, "lw": 0.8, "zorder": 1}
    ax.plot([-L / 2, L / 2, L / 2, -L / 2, -L / 2], [-W / 2, -W / 2, W / 2, W / 2, -W / 2], **kw)
    ax.plot([0, 0], [-W / 2, W / 2], **kw)
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, **kw))
    for s in (-1, 1):
        x0 = s * L / 2
        ax.plot([x0, x0 - s * 16.5, x0 - s * 16.5, x0], [-20.16, -20.16, 20.16, 20.16], **kw)
    ax.set_xlim(-L / 2 - 3, L / 2 + 3)
    ax.set_ylim(-W / 2 - 3, W / 2 + 3)
    ax.set_aspect("equal")
    ax.axis("off")


def figure_cameras(match: str = "J03WPY", frame_idx: int | None = None) -> None:
    """Fig. 1: the same frame under the rectangle camera and the fitted wedge camera."""
    from pitchlens.analytics.pressing import prepare_tracking
    from pitchlens.data.datasets import load_tracking
    from pitchlens.simulation.camera import BroadcastCamera, WedgeCamera
    from pitchlens.simulation.occlusion import apply_broadcast_view, wedge_path

    ds = load_tracking(match, limit=4000)
    _, data = prepare_tracking(ds)
    params = json.loads(Path("configs/cameras/wedge_skillcorner.json").read_text())
    wedge = WedgeCamera(**params)
    rect = BroadcastCamera(view_length=40)
    keys = ["period_id", "frame_id", "id"]
    if frame_idx is None:  # the frame where the two cameras disagree most
        a = apply_broadcast_view(data, rect).select(keys).with_columns(pl.lit(1).alias("r"))
        b = apply_broadcast_view(data, wedge).select(keys).with_columns(pl.lit(1).alias("w"))
        diff = (
            data.filter(pl.col("id") != "ball").select(keys)
            .join(a, on=keys, how="left").join(b, on=keys, how="left")
            .group_by("period_id", "frame_id")
            .agg((pl.col("r").is_null() != pl.col("w").is_null()).sum().alias("d"))
            .sort("d", "frame_id", descending=[True, False])
        )
        key = diff.row(0, named=True)
    else:
        frames = data.select("period_id", "frame_id").unique().sort("period_id", "frame_id")
        key = frames.row(frame_idx, named=True)
    one = data.filter(
        (pl.col("period_id") == key["period_id"]) & (pl.col("frame_id") == key["frame_id"])
    )
    teams = sorted(one.filter(pl.col("id") != "ball")["team_id"].unique().to_list())
    team_color = dict(zip(teams, (BLUE, ORANGE)))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, cam, title in (
        (axes[0], rect, "Rectangle camera (40 m, Exp. 1)"),
        (axes[1], wedge, "Camera fitted to broadcasts (Exp. 3)"),
    ):
        kept = apply_broadcast_view(data, cam)
        kept_ids = set(
            kept.filter(
                (pl.col("period_id") == key["period_id"]) & (pl.col("frame_id") == key["frame_id"])
            )["id"].to_list()
        )
        _pitch(ax)
        ball = one.filter(pl.col("id") == "ball").row(0, named=True)
        if isinstance(cam, BroadcastCamera):
            x0, x1, y0, y1 = cam.window(ball["x"] + 52.5, ball["y"] + 34)
            ax.add_patch(plt.Rectangle((x0 - 52.5, y0 - 34), x1 - x0, y1 - y0,
                                       color=BLUE, alpha=0.10, lw=0, zorder=0))  # fmt: skip
        else:
            path = wedge_path(data, cam).filter(
                (pl.col("period_id") == key["period_id"]) & (pl.col("frame_id") == key["frame_id"])
            ).row(0, named=True)
            pan, hf, near = path["pan"], path["half_fov"], path["near"]
            far = 200.0
            pts = []
            for a in (pan - hf, pan + hf):
                for d in (near / np.cos(a - pan), far):
                    pts.append((cam.cam_x + d * np.sin(a), cam.cam_y + d * np.cos(a)))
            poly = [pts[0], pts[1], pts[3], pts[2]]
            ax.add_patch(plt.Polygon(poly, color=BLUE, alpha=0.10, lw=0, zorder=0))
        for r in one.filter(pl.col("id") != "ball").iter_rows(named=True):
            vis = r["id"] in kept_ids
            c = team_color.get(r["team_id"], INK2)
            ax.scatter(r["x"], r["y"], s=26, zorder=3, linewidths=1.2,
                       facecolor=c if vis else "white", edgecolor=c if vis else HIDDEN)  # fmt: skip
        ax.scatter(ball["x"], ball["y"], s=12, color=INK, zorder=4)
        n_vis = len(kept_ids - {"ball"})
        label = f"{title}\n{n_vis} of 22 players visible"
        ax.set_title(label, fontsize=8, color=INK, pad=4)
    fig.text(0.5, 0.01, "Filled: on screen  ·  hollow grey: hidden  ·  shaded: camera footprint "
             f"(IDSSE {match}, one frame)", ha="center", fontsize=7, color=INK2)  # fmt: skip
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(OUT / "fig1_cameras.pdf")
    plt.close(fig)


def _per_camera() -> pl.DataFrame:
    lag = {
        f"st_{f.stem}": json.loads(f.read_text())["lag_s"]
        for f in Path("configs/cameras/stadiums").glob("*.json")
    }
    parts = []
    for d, name in (("idsse", "IDSSE (7 matches)"), ("metrica", "Metrica (2 matches)")):
        t = pl.read_csv(f"outputs/stadiums/{d}/stadium_per_camera.csv")
        parts.append(t.with_columns(pl.lit(name).alias("data"),
                                    pl.col("camera").replace_strict(lag).alias("lag_s")))  # fmt: skip
    return pl.concat(parts)


def figure_visibility(t: pl.DataFrame) -> None:
    """Fig. 2: active-presser bias vs. visible players, one point per stadium camera."""
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for (name,), part in t.group_by("data", maintain_order=True):
        color, marker = (BLUE, "o") if name.startswith("IDSSE") else (ORANGE, "s")
        x, y = part["players_visible"].to_numpy(), part["active_bias"].to_numpy()
        ax.scatter(x, y, s=22, marker=marker, facecolor=color, edgecolor="white",
                   linewidths=0.8, zorder=3, label=name)  # fmt: skip
        k, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 2)
        ax.plot(xs, k * xs + b, color=color, lw=1.5, zorder=2)
        ax.annotate(f"{k:+.1f} pp / player", (xs[-1], k * xs[-1] + b), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7, color=INK2)  # fmt: skip
    ax.set_xlabel("Players visible per frame (stadium camera)")
    ax.set_ylabel("Bias in active pressers (%)")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlim(right=ax.get_xlim()[1] + 1.2)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_visibility.pdf")
    plt.close(fig)


def figure_lag(t: pl.DataFrame) -> None:
    """Fig. 3: bias in pressure on the ball carrier vs. fitted camera lag."""
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    offsets = {"IDSSE (7 matches)": -0.012, "Metrica (2 matches)": 0.012}
    for (name,), part in t.group_by("data", maintain_order=True):
        color, marker = (BLUE, "o") if name.startswith("IDSSE") else (ORANGE, "s")
        rng = np.random.default_rng(0)
        x = part["lag_s"].to_numpy() + offsets[name] + rng.uniform(-0.008, 0.008, part.height)
        ax.scatter(x, part["pressure_bias"], s=22, marker=marker, facecolor=color,
                   edgecolor="white", linewidths=0.8, zorder=3, label=name)  # fmt: skip
    means = t.group_by("lag_s").agg(pl.col("pressure_bias").mean(), pl.len()).sort("lag_s")
    ax.plot(means["lag_s"], means["pressure_bias"], color=INK2, lw=1.2, zorder=2)
    for r in means.iter_rows(named=True):
        ax.annotate(f"n={r['len'] // 2}", (r["lag_s"], 0.05), ha="center", fontsize=6.5,
                    color=INK2, annotation_clip=False)  # fmt: skip
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_xlabel("Fitted camera lag (s)")
    ax.set_ylabel("Bias in pressure on ball carrier (%)")
    ax.set_ylim(top=0.4)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_lag.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    per_camera = _per_camera()
    figure_visibility(per_camera)
    figure_lag(per_camera)
    figure_cameras()
    print("Saved figures to", OUT)
