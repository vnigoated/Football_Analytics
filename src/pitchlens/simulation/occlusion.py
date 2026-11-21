"""Hide players the virtual broadcast camera would not see (Experiment 1, RQ1).

Works on the long-format tracking frame used by ``unravelsports`` (one row per
object per frame, ball rows have ``id == "ball"``, coordinates centred on the pitch).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pitchlens.simulation.camera import BroadcastCamera

KEYS = ["period_id", "frame_id"]


def camera_centres(
    tracking: pl.DataFrame,
    camera: BroadcastCamera,
    centred: bool = True,
) -> pl.DataFrame:
    """Per-frame camera window ``(x0, x1, y0, y1)`` in the tracking's coordinate system.

    The camera follows the ball (with ``camera.smoothing`` lag, reset every period).
    """
    ox, oy = (camera.pitch_length / 2, camera.pitch_width / 2) if centred else (0.0, 0.0)
    ball = (
        tracking.filter(pl.col("id") == "ball")
        .select(*KEYS, "x", "y")
        .unique(subset=KEYS, keep="first")
        .sort(KEYS)
    )
    parts = []
    for period, part in ball.group_by("period_id", maintain_order=True):
        xy = part.select("x", "y").to_numpy().astype(np.float64) + [ox, oy]
        xy = np.nan_to_num(xy, nan=0.0)
        centre = camera.track_ball(xy)
        win = np.array([camera.window(cx, cy) for cx, cy in centre])
        # A window edge lying on a pitch line also shows what is beyond it (throw-ins,
        # keepers behind the goal line), so open that edge.
        eps = 1e-6
        win[win[:, 0] <= eps, 0] = -np.inf
        win[win[:, 1] >= camera.pitch_length - eps, 1] = np.inf
        win[win[:, 2] <= eps, 2] = -np.inf
        win[win[:, 3] >= camera.pitch_width - eps, 3] = np.inf
        win -= [ox, ox, oy, oy]
        parts.append(
            part.select(KEYS).with_columns(
                pl.Series("x0", win[:, 0]),
                pl.Series("x1", win[:, 1]),
                pl.Series("y0", win[:, 2]),
                pl.Series("y1", win[:, 3]),
            )
        )
    return pl.concat(parts)


def apply_broadcast_view(
    tracking: pl.DataFrame,
    camera: BroadcastCamera,
    centred: bool = True,
    keep_carrier: bool = True,
) -> pl.DataFrame:
    """Drop player rows outside the camera window. Ball rows are always kept.

    ``keep_carrier`` keeps the ball carrier even if the clipped window misses them
    (a broadcast camera essentially always shows the player on the ball).
    """
    win = camera_centres(tracking, camera, centred=centred)
    inside = (
        (pl.col("x") >= pl.col("x0"))
        & (pl.col("x") <= pl.col("x1"))
        & (pl.col("y") >= pl.col("y0"))
        & (pl.col("y") <= pl.col("y1"))
    )
    keep = inside | (pl.col("id") == "ball")
    if keep_carrier:
        keep = keep | pl.col("is_ball_carrier")
    return tracking.join(win, on=KEYS, how="left").filter(keep).drop("x0", "x1", "y0", "y1")


def visibility_stats(full: pl.DataFrame, masked: pl.DataFrame) -> dict[str, float]:
    """Average number of visible players per frame and share of all player rows kept."""
    def per_frame(df: pl.DataFrame) -> pl.Series:
        return df.filter(pl.col("id") != "ball").group_by(KEYS).len()["len"]

    n_full, n_vis = per_frame(full), per_frame(masked)
    return {
        "players_per_frame_full": float(n_full.mean()),
        "players_per_frame_visible": float(n_vis.mean()),
        "share_visible": float(n_vis.sum() / n_full.sum()),
    }
