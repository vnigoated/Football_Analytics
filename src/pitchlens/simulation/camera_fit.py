"""Fit the WedgeCamera to real broadcast footprints (SkillCorner) and validate it.

For every frame SkillCorner gives the camera footprint on the pitch (4 corners) and,
per player, whether they were on screen (``is_detected``). Geometry used here:

- the left and right image edges are rays from the camera's ground position, so their
  intersection locates the camera (median over a match);
- pan = mean bearing of the two far corners, fov = angle between them;
- near = depth of the bottom edge along the viewing direction.

Dynamics are then fitted as functions of the ball (bearing, distance) with least squares,
the lag by a grid search, and the leftover variation as AR(1) noise.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pitchlens.simulation.camera import WedgeCamera

FPS = 10.0  # SkillCorner frame rate


def _segments(frame: np.ndarray, period: np.ndarray) -> np.ndarray:
    """Id of each run of consecutive frames within a period."""
    brk = np.r_[False, (np.diff(frame) != 1) | (np.diff(period) != 0)]
    return np.cumsum(brk)


def _ema_seg(x: np.ndarray, alpha: float, seg: np.ndarray) -> np.ndarray:
    out = x.astype(np.float64).copy()
    for i in range(1, len(x)):
        if seg[i] == seg[i - 1]:
            out[i] = alpha * out[i - 1] + (1 - alpha) * x[i]
    return out


def camera_position(frames: pl.DataFrame) -> tuple[float, float]:
    """Median ground position of the camera from the intersection of the image side edges."""
    f = frames
    bl = f.select("x_bottom_left", "y_bottom_left").to_numpy()
    tl = f.select("x_top_left", "y_top_left").to_numpy()
    br = f.select("x_bottom_right", "y_bottom_right").to_numpy()
    tr = f.select("x_top_right", "y_top_right").to_numpy()
    d1, d2 = tl - bl, tr - br
    den = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((br[:, 0] - bl[:, 0]) * d2[:, 1] - (br[:, 1] - bl[:, 1]) * d2[:, 0]) / den
    cam = bl + t[:, None] * d1
    ok = np.isfinite(cam).all(1) & (np.abs(den) > 1e-3) & (cam[:, 1] < -34)
    return float(np.median(cam[ok, 0])), float(np.median(cam[ok, 1]))


def footprint_series(
    frames: pl.DataFrame, cam: tuple[float, float], players: pl.DataFrame | None = None
) -> dict[str, np.ndarray]:
    """Per-frame pan, fov, near, ball bearing/distance and players' bearing stats."""
    cx, cy = cam
    f = frames.sort("period", "frame")
    if players is not None:
        b = pl.arctan2(pl.col("x") - cx, pl.col("y") - cy)
        stats = players.group_by("frame").agg(b.mean().alias("pb"), b.std().alias("ps"))
        f = f.join(stats, on="frame", how="left").sort("period", "frame")
    has = "pb" in f.columns

    def bearing(x, y):
        return np.arctan2(f[x].to_numpy() - cx, f[y].to_numpy() - cy)

    left, right = bearing("x_top_left", "y_top_left"), bearing("x_top_right", "y_top_right")
    pan = (left + right) / 2
    mid = np.c_[
        (f["x_bottom_left"].to_numpy() + f["x_bottom_right"].to_numpy()) / 2 - cx,
        (f["y_bottom_left"].to_numpy() + f["y_bottom_right"].to_numpy()) / 2 - cy,
    ]
    bx, by = f["ball_x"].to_numpy() - cx, f["ball_y"].to_numpy() - cy
    return {
        "pan": pan,
        "fov": right - left,
        "near": mid[:, 0] * np.sin(pan) + mid[:, 1] * np.cos(pan),
        "ball_bearing": np.arctan2(bx, by),
        "ball_dist": np.hypot(bx, by),
        "ball_detected": f["ball_detected"].to_numpy(),
        "players_bearing": f["pb"].fill_null(0).to_numpy() if has else np.zeros(f.height),
        "players_spread": f["ps"].fill_null(0).to_numpy() if has else np.zeros(f.height),
        "seg": _segments(f["frame"].to_numpy(), f["period"].to_numpy()),
    }


def _ar_tau(resid: np.ndarray, seg: np.ndarray, lag_frames: int = 10) -> float:
    """Time constant (s) of an AR(1) process from the autocorrelation at ``lag_frames``."""
    same = seg[lag_frames:] == seg[:-lag_frames]
    r = np.corrcoef(resid[lag_frames:][same], resid[:-lag_frames][same])[0, 1]
    r = float(np.clip(r, 0.01, 0.999))
    return -(lag_frames / FPS) / np.log(r)


def fit_wedge_camera(
    match_frames: list[pl.DataFrame],
    match_players: list[pl.DataFrame] | None = None,
) -> tuple[WedgeCamera, dict]:
    """Fit one WedgeCamera to several matches. Returns the camera and a fit report.

    With ``match_players`` the pan also follows the players' centre and the zoom their
    spread (an operator frames the play, not only the ball).
    """
    use_players = match_players is not None
    cams, series = [], []
    for i, f in enumerate(match_frames):
        cam = camera_position(f)
        cams.append(cam)
        s = footprint_series(f, cam, match_players[i] if use_players else None)
        s["seg"] = s["seg"] + (series[-1]["seg"].max() + 1 if series else 0)
        series.append(s)
    s = {k: np.concatenate([x[k] for x in series]) for k in series[0]}
    det, seg = s["ball_detected"], s["seg"]

    # pan = gain * bearing + offset (ball seen), then the lag
    A = np.c_[s["ball_bearing"], s["players_bearing"], np.ones(len(det))]
    cols = [0, 1, 2] if use_players else [0, 2]
    coef = np.zeros(3)
    coef[cols] = np.linalg.lstsq(A[det][:, cols], s["pan"][det], rcond=None)[0]
    gain, gain_players, offset = coef
    target = A @ coef
    lags = np.round(np.arange(0, 1.01, 0.1), 1)
    rmse = []
    for lag in lags:
        a = float(np.exp(-1 / (FPS * lag))) if lag > 0 else 0.0
        e = _ema_seg(target, a, seg) - s["pan"]
        rmse.append(float(np.sqrt(np.mean(e[det] ** 2))))
    lag_s = float(lags[int(np.argmin(rmse))])
    a = float(np.exp(-1 / (FPS * lag_s))) if lag_s > 0 else 0.0
    pan_resid = s["pan"] - _ema_seg(target, a, seg)

    B = np.c_[np.ones(len(det)), s["ball_dist"], s["players_spread"]]
    fcols = [0, 1, 2] if use_players else [0, 1]
    fcoef = np.zeros(3)
    fcoef[fcols] = np.linalg.lstsq(B[:, fcols], s["fov"], rcond=None)[0]
    fov_a, fov_b, fov_c = fcoef
    near_a, near_b = np.linalg.lstsq(B[:, :2], s["near"], rcond=None)[0]
    fov_resid = s["fov"] - B @ fcoef
    near_resid = s["near"] - (near_a + near_b * s["ball_dist"])
    tau = float(np.median([_ar_tau(r, seg) for r in (pan_resid, fov_resid, near_resid)]))

    cam_xy = np.median(np.array(cams), axis=0)
    camera = WedgeCamera(
        cam_x=float(cam_xy[0]),
        cam_y=float(cam_xy[1]),
        pan_gain=float(gain),
        pan_gain_players=float(gain_players),
        pan_offset=float(offset),
        lag_s=lag_s,
        fov_a=float(fov_a),
        fov_b=float(fov_b),
        fov_c=float(fov_c),
        fov_min=float(np.percentile(s["fov"], 1)),
        fov_max=float(np.percentile(s["fov"], 99)),
        near_a=float(near_a),
        near_b=float(near_b),
        pan_noise_sd=float(np.std(pan_resid)),
        fov_noise_sd=float(np.std(fov_resid)),
        near_noise_sd=float(np.std(near_resid)),
        noise_tau_s=tau,
    )
    report = {
        "matches": len(match_frames),
        "frames": len(det),
        "camera_positions": [tuple(round(v, 1) for v in c) for c in cams],
        "lag_rmse_deg": {float(lg): round(float(np.degrees(r)), 2) for lg, r in zip(lags, rmse)},
        "fov_deg_p10_p50_p90": np.round(np.degrees(np.percentile(s["fov"], [10, 50, 90])), 1).tolist(),
        "r2_fov": float(1 - np.var(fov_resid) / np.var(s["fov"])),
        "r2_near": float(1 - np.var(near_resid) / np.var(s["near"])),
        "r2_pan": float(1 - np.var(pan_resid) / np.var(s["pan"])),
        "uses_players": use_players,
    }
    return camera, report


def skillcorner_long(frames: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    """SkillCorner in the long format used by ``apply_broadcast_view`` (plus ``is_detected``)."""
    ball = frames.select(
        pl.col("period").alias("period_id"),
        pl.col("frame").alias("frame_id"),
        pl.lit("ball").alias("id"),
        pl.col("ball_x").alias("x"),
        pl.col("ball_y").alias("y"),
        pl.lit(True).alias("is_detected"),
    )
    pl_rows = players.join(frames.select("frame", "period"), on="frame", how="inner").select(
        pl.col("period").alias("period_id"),
        pl.col("frame").alias("frame_id"),
        pl.col("player_id").cast(pl.Utf8).alias("id"),
        "x",
        "y",
        "is_detected",
    )
    return pl.concat([ball, pl_rows]).with_columns(pl.lit(False).alias("is_ball_carrier"))


def footprint_visibility(frames: pl.DataFrame, long: pl.DataFrame) -> np.ndarray:
    """Upper bound: is each player row inside the real footprint polygon?"""
    df = long.filter(pl.col("id") != "ball").join(
        frames.select(pl.col("period").alias("period_id"), pl.col("frame").alias("frame_id"),
                      *[c for c in frames.columns if c.startswith(("x_", "y_"))]),
        on=["period_id", "frame_id"], how="left",
    )  # fmt: skip
    px, py = df["x"].to_numpy(), df["y"].to_numpy()
    quad = [("bottom_left", "top_left"), ("top_left", "top_right"),
            ("top_right", "bottom_right"), ("bottom_right", "bottom_left")]  # fmt: skip
    signs = []
    for a, b in quad:
        ax, ay = df[f"x_{a}"].to_numpy(), df[f"y_{a}"].to_numpy()
        bx, by = df[f"x_{b}"].to_numpy(), df[f"y_{b}"].to_numpy()
        signs.append((bx - ax) * (py - ay) - (by - ay) * (px - ax))
    signs = np.array(signs)
    return (signs >= 0).all(0) | (signs <= 0).all(0)


def evaluate_camera(camera, frames: pl.DataFrame, players: pl.DataFrame) -> dict[str, float]:
    """Compare a camera's predicted visibility with SkillCorner's ``is_detected``."""
    from pitchlens.simulation.occlusion import apply_broadcast_view

    long = skillcorner_long(frames, players)
    kept = apply_broadcast_view(long, camera, keep_carrier=False)
    rows = long.filter(pl.col("id") != "ball").with_row_index("_r")
    vis = rows.join(kept.select("period_id", "frame_id", "id").with_columns(pl.lit(True).alias("_v")),
                    on=["period_id", "frame_id", "id"], how="left")["_v"].fill_null(False).to_numpy()  # fmt: skip
    det = rows["is_detected"].to_numpy()
    n_frames = rows.select("period_id", "frame_id").unique().height
    return {
        "visible_per_frame_pred": float(vis.sum() / n_frames),
        "visible_per_frame_true": float(det.sum() / n_frames),
        "accuracy": float((vis == det).mean()),
        "recall_hidden": float((~vis & ~det).sum() / max((~det).sum(), 1)),
    }
