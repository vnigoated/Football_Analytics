"""Hide players the virtual broadcast camera would not see (Experiment 1, RQ1).

Works on the long-format tracking frame used by ``unravelsports`` (one row per
object per frame, ball rows have ``id == "ball"``, coordinates centred on the pitch).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pitchlens.simulation.camera import BroadcastCamera, WedgeCamera

KEYS = ["period_id", "frame_id"]


def camera_path(
    tracking: pl.DataFrame,
    camera: BroadcastCamera,
    centred: bool = True,
) -> pl.DataFrame:
    """Per-frame camera centre ``cx`` and half-length ``hl`` (pitch coordinates, 0..L),
    plus the y bounds ``y0``/``y1`` in the tracking's coordinate system.

    The camera follows the ball (lag reset every period). Missing ball positions are
    filled forward, so the camera holds still while the ball is lost.
    """
    ox, oy = (camera.pitch_length / 2, camera.pitch_width / 2) if centred else (0.0, 0.0)
    ball = (
        tracking.filter(pl.col("id") == "ball")
        .select(*KEYS, "x", "y")
        .unique(subset=KEYS, keep="first")
        .sort(KEYS)
        .with_columns(
            pl.col("x", "y").fill_nan(None).forward_fill().backward_fill().over("period_id")
        )
    )
    eps = 1e-6
    parts = []
    for _, part in ball.group_by("period_id", maintain_order=True):
        xy = np.nan_to_num(part.select("x", "y").to_numpy().astype(np.float64)) + [ox, oy]
        st = camera.path(xy)
        y0 = st["cy"] - st["half_width"]
        y1 = st["cy"] + st["half_width"]
        # A window edge on a touchline also shows what is beyond it, so open it.
        y0 = np.where(y0 <= eps, -np.inf, y0 - oy)
        y1 = np.where(y1 >= camera.pitch_width - eps, np.inf, y1 - oy)
        parts.append(
            part.select(KEYS).with_columns(
                pl.Series("cx", st["cx"]),
                pl.Series("hl", st["half_length"]),
                pl.Series("y0", y0),
                pl.Series("y1", y1),
            )
        )
    return pl.concat(parts)


def wedge_path(tracking: pl.DataFrame, camera: WedgeCamera) -> pl.DataFrame:
    """Per-frame ``pan``, ``half_fov`` and ``near`` of a WedgeCamera (centred coordinates)."""
    rng = np.random.default_rng(camera.seed) if camera.seed is not None else None
    ball = (
        tracking.filter(pl.col("id") == "ball")
        .select(*KEYS, "x", "y")
        .unique(subset=KEYS, keep="first")
        .sort(KEYS)
        .with_columns(
            pl.col("x", "y").fill_nan(None).forward_fill().backward_fill().over("period_id")
        )
    )
    bearing = pl.arctan2(pl.col("x") - camera.cam_x, pl.col("y") - camera.cam_y)
    players = (
        tracking.filter(pl.col("id") != "ball")
        .group_by(KEYS)
        .agg(bearing.mean().alias("pb"), bearing.std().alias("ps"))
    )
    ball = ball.join(players, on=KEYS, how="left").sort(KEYS)
    parts = []
    for _, part in ball.group_by("period_id", maintain_order=True):
        st = camera.path(
            np.nan_to_num(part.select("x", "y").to_numpy().astype(np.float64)),
            rng,
            players_bearing=part["pb"].to_numpy(),
            players_spread=part["ps"].to_numpy(),
        )
        parts.append(
            part.select(KEYS).with_columns(
                pl.Series("pan", st["pan"]),
                pl.Series("half_fov", st["half_fov"]),
                pl.Series("near", st["near"]),
            )
        )
    return pl.concat(parts)


def _apply_wedge(tracking: pl.DataFrame, camera: WedgeCamera, keep_carrier: bool) -> pl.DataFrame:
    dx, dy = pl.col("x") - camera.cam_x, pl.col("y") - camera.cam_y
    d = pl.arctan2(dx, dy) - pl.col("pan")
    rel = pl.arctan2(d.sin(), d.cos())  # wrapped to (-pi, pi]
    depth = dx * pl.col("pan").sin() + dy * pl.col("pan").cos()
    keep = ((rel.abs() <= pl.col("half_fov")) & (depth >= pl.col("near"))) | (pl.col("id") == "ball")
    if keep_carrier and "is_ball_carrier" in tracking.columns:
        keep = keep | pl.col("is_ball_carrier")
    path = wedge_path(tracking, camera)
    return tracking.join(path, on=KEYS, how="left").filter(keep).drop("pan", "half_fov", "near")


def apply_broadcast_view(
    tracking: pl.DataFrame,
    camera: BroadcastCamera | WedgeCamera,
    centred: bool = True,
    keep_carrier: bool = True,
) -> pl.DataFrame:
    """Drop player rows outside the camera footprint. Ball rows are always kept.

    ``keep_carrier`` keeps the ball carrier even if the (lagging) camera misses them
    (a broadcast camera essentially always shows the player on the ball).
    """
    if isinstance(camera, WedgeCamera):
        if not centred:
            raise ValueError("WedgeCamera expects centred coordinates")
        return _apply_wedge(tracking, camera, keep_carrier)
    ox, oy = (camera.pitch_length / 2, camera.pitch_width / 2) if centred else (0.0, 0.0)
    L, W, eps = camera.pitch_length, camera.pitch_width, 1e-6
    path = camera_path(tracking, camera, centred=centred)
    px, py = pl.col("x") + ox, pl.col("y") + oy  # pitch coordinates
    # perspective: shorter footprint on the near touchline (y=0), longer on the far one
    hl = pl.col("hl") * (1 + camera.perspective * (2 * py.clip(0, W) / W - 1))
    x0, x1 = pl.col("cx") - hl, pl.col("cx") + hl
    # an edge that reaches a goal line also shows what is behind it
    in_x = ((x0 <= eps) | (px >= x0)) & ((x1 >= L - eps) | (px <= x1))
    in_y = (pl.col("y") >= pl.col("y0")) & (pl.col("y") <= pl.col("y1"))
    keep = (in_x & in_y) | (pl.col("id") == "ball")
    if keep_carrier:
        keep = keep | pl.col("is_ball_carrier")
    return tracking.join(path, on=KEYS, how="left").filter(keep).drop("cx", "hl", "y0", "y1")


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
