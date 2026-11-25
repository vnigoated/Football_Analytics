"""Real broadcast camera footprints from SoccerNet-GSR (journal revision, step 3).

SN-GSR gives no camera parameters, but the labelled pitch positions of the people (``bbox_pitch_raw``) are an exact
projection of their box bottoms (homography residual 0.002 m, A5). A homography fitted to them per frame maps the four
image corners onto the pitch: the camera's footprint, as SkillCorner provides it. With the ball position this is the
input of ``pitchlens.simulation.camera_fit``, so the same camera model can be fitted to a second league (Swiss Super
League) and compared with the A-League fit.

Coordinates: metres, origin at the pitch centre; the footprint is rotated by 180° if needed so that the camera stands
behind the touchline at negative y, as in SkillCorner.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def _project(H: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Projected points and the homogeneous scale (<= 0: above the horizon)."""
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3], p[:, 2]


def clip_footprints(images: pl.DataFrame, objects: pl.DataFrame, width: int = 1920, height: int = 1080,
                    max_extent: float = 200.0) -> pl.DataFrame:
    """Per labelled frame of one clip: the four image corners on the pitch and the labelled ball position.

    Columns: ``frame, x_top_left, y_top_left, ... , x_bottom_right, y_bottom_right, ball_x, ball_y, ball_detected``.
    Frames without a homography (fewer than 6 labelled people) or with a corner above the horizon or further than
    ``max_extent`` metres are skipped.
    """
    from pitchlens.vision.keypoints import frame_homography

    img_pts = np.array([[0, 0], [width, 0], [0, height], [width, height]], float)
    names = ["top_left", "top_right", "bottom_left", "bottom_right"]
    by_image = {k[0]: g for k, g in objects.group_by("image_id")}
    rows = []
    for r in images.iter_rows(named=True):
        g = by_image.get(r["image_id"])
        if g is None:
            continue
        H, _, _ = frame_homography(g, use_raw=True)
        if H is None:
            continue
        xy, w = _project(H, img_pts)
        if (w <= 0).any() or not np.isfinite(xy).all() or np.abs(xy).max() > max_extent:
            continue
        ball = g.filter((pl.col("role") == "ball") & pl.col("pitch_raw_x").is_not_null())
        row = {"frame": r["frame"]}
        for n, (x, y) in zip(names, xy, strict=True):
            row[f"x_{n}"], row[f"y_{n}"] = float(x), float(y)
        row["ball_x"] = float(ball["pitch_raw_x"][0]) if ball.height else None
        row["ball_y"] = float(ball["pitch_raw_y"][0]) if ball.height else None
        rows.append(row)
    schema = {"frame": pl.Int64, **{f"{a}_{n}": pl.Float64 for n in names for a in ("x", "y")},
              "ball_x": pl.Float64, "ball_y": pl.Float64}
    df = pl.DataFrame(rows, schema=schema).sort("frame")
    return df.with_columns(ball_detected=pl.col("ball_x").is_not_null()).with_columns(
        # unlabelled ball: interpolate within the clip, then hold at the ends; else the footprint centre
        pl.col("ball_x").interpolate().forward_fill().backward_fill(),
        pl.col("ball_y").interpolate().forward_fill().backward_fill(),
    ).with_columns(
        pl.col("ball_x").fill_null((pl.col("x_bottom_left") + pl.col("x_bottom_right")) / 2),
        pl.col("ball_y").fill_null((pl.col("y_bottom_left") + pl.col("y_top_left")) / 2),
    )


def orient(frames: pl.DataFrame) -> pl.DataFrame:
    """Put the camera at negative y looking towards +y with the image's left edge at smaller x (SkillCorner's
    convention): flip y if the near edge is at larger y than the far edge, then flip x if the image's left corner is
    at larger x than its right corner."""
    f = frames
    if f.is_empty():
        return f
    near_y = ((f["y_bottom_left"] + f["y_bottom_right"]) / 2).median()
    far_y = ((f["y_top_left"] + f["y_top_right"]) / 2).median()
    if near_y is not None and far_y is not None and near_y > far_y:
        f = f.with_columns([(-pl.col(c)).alias(c) for c in f.columns if c.startswith("y_") or c == "ball_y"])
    if (f["x_top_left"] - f["x_top_right"]).median() > 0:
        f = f.with_columns([(-pl.col(c)).alias(c) for c in f.columns if c.startswith("x_") or c == "ball_x"])
    return f


def visible_players(images: pl.DataFrame, objects: pl.DataFrame) -> float:
    """Mean labelled players + goalkeepers per labelled frame (the real visibility of the clip)."""
    from pitchlens.data.gsr import visible_per_frame

    v = visible_per_frame(images, objects)
    return float(v["players"].mean()) if v.height else float("nan")
