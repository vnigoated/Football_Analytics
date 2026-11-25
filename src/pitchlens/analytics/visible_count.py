"""How well the video pipeline counts visible players (Phase 2 · B1).

The Phase 1 correction divides a broadcast's pressing value by ``1 + bias``, with ``bias = k · visible + b`` and
``visible`` the mean number of players (both teams, goalkeepers included, no referees) per frame. On real video
``visible`` must come from the pipeline, so its error matters: an error of δ players in the broadcast mean shifts the
predicted bias by ``k · δ`` percentage points (k ≈ 4.1 pp per player, paper v1).

Tables (polars), one row per labelled frame: ``sequence, frame, gt, ours`` (+ context columns such as ``zoom``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import polars as pl

SLOPE_PP_PER_PLAYER = 4.1  # paper v1: pooled slope of active-presser bias against visible players
COUNT_ROLES = ("player", "goalkeeper")


def gt_counts(images: pl.DataFrame, objects: pl.DataFrame) -> pl.DataFrame:
    """Labelled players + goalkeepers per labelled frame (A1's definition), with a zoom proxy: the median height (px)
    of the labelled player boxes in the frame (small = wide view)."""
    from pitchlens.data.gsr import visible_per_frame

    v = visible_per_frame(images, objects).select("sequence", "image_id", "frame", pl.col("players").alias("gt"))
    zoom = (objects.filter(pl.col("role").is_in(COUNT_ROLES) & pl.col("h").is_not_null())
            .group_by("image_id").agg(pl.col("h").median().alias("zoom")))
    return v.join(zoom, on="image_id", how="left").drop("image_id")


def predicted_counts(tracks: pl.DataFrame, roles: dict[int, tuple[str, str | None]],
                     frames: Iterable[int]) -> pl.DataFrame:
    """Our players + goalkeepers per frame: tracks whose role is player or goalkeeper (no homography needed)."""
    keep = [t for t, (r, _) in roles.items() if r in COUNT_ROLES]
    c = tracks.filter(pl.col("id").is_in(keep)).group_by("frame").len().rename({"len": "ours"})
    return (pl.DataFrame({"frame": list(frames)}, schema={"frame": pl.Int64}).join(c, on="frame", how="left")
            .with_columns(pl.col("ours").fill_null(0).cast(pl.Int64)))


def pitch_counts(pred: pl.DataFrame, frames: Iterable[int]) -> pl.DataFrame:
    """Players + goalkeepers per frame that also have a pitch position (the game-state output; 0 without a
    homography)."""
    c = pred.filter(pl.col("role").is_in(COUNT_ROLES)).group_by("frame").len().rename({"len": "ours"})
    return (pl.DataFrame({"frame": list(frames)}, schema={"frame": pl.Int64}).join(c, on="frame", how="left")
            .with_columns(pl.col("ours").fill_null(0).cast(pl.Int64)))


def fit_calibration(df: pl.DataFrame) -> tuple[float, float]:
    """Least-squares line ``gt ≈ a · ours + b`` over frames (fit on the tuning half only)."""
    a, b = np.polyfit(df["ours"].to_numpy().astype(float), df["gt"].to_numpy().astype(float), 1)
    return float(a), float(b)


def calibrate(df: pl.DataFrame, a: float, b: float, column: str = "ours") -> pl.DataFrame:
    return df.with_columns((pl.col(column) * a + b).alias(column))


def clip_bootstrap(df: pl.DataFrame, stat: Callable[[pl.DataFrame], float], n_boot: int = 2000,
                   seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """(estimate, low, high): ``stat`` on all frames, with a percentile interval from resampling whole clips."""
    rng = np.random.default_rng(seed)
    parts = {k[0]: g for k, g in df.group_by("sequence")}
    names = list(parts)
    est = stat(df)
    boots = [stat(pl.concat([parts[names[i]] for i in rng.integers(len(names), size=len(names))]))
             for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(est), float(lo), float(hi)


def count_summary(df: pl.DataFrame, slope: float = SLOPE_PP_PER_PLAYER, n_boot: int = 1000) -> dict[str, float]:
    """Frame and clip level errors of ``ours`` against ``gt``, and what the clip-level error would do to the
    corrected pressing value (``slope`` pp per player)."""
    d = df.with_columns(diff=pl.col("ours").cast(pl.Float64) - pl.col("gt").cast(pl.Float64))
    clips = d.group_by("sequence").agg(pl.col("gt").mean(), pl.col("ours").mean(), pl.col("diff").mean())
    bias, lo, hi = clip_bootstrap(d, lambda x: float(x["diff"].mean()), n_boot=n_boot)
    ce = clips["diff"].to_numpy()
    return {
        "clips": clips.height,
        "frames": d.height,
        "gt_mean": round(float(d["gt"].mean()), 2),
        "ours_mean": round(float(d["ours"].mean()), 2),
        "bias": round(bias, 2),
        "bias_ci_low": round(lo, 2),
        "bias_ci_high": round(hi, 2),
        "frame_mae": round(float(d["diff"].abs().mean()), 2),
        "frame_within_1": round(float((d["diff"].abs() <= 1).mean()), 3),
        "frame_within_2": round(float((d["diff"].abs() <= 2).mean()), 3),
        "clip_mae": round(float(np.abs(ce).mean()), 2),
        "clip_p95_abs": round(float(np.quantile(np.abs(ce), 0.95)), 2),
        "clip_r": round(float(np.corrcoef(clips["gt"], clips["ours"])[0, 1]), 3) if clips.height > 2 else float("nan"),
        "implied_bias_shift_pp": round(slope * bias, 2),
        "implied_clip_p95_pp": round(slope * float(np.quantile(np.abs(ce), 0.95)), 2),
    }


def error_by(df: pl.DataFrame, column: str, edges: list[float] | None = None, labels: list[str] | None = None,
             ) -> pl.DataFrame:
    """Mean count error, absolute error and frame share per group of ``column`` (binned at ``edges`` if given)."""
    d = df.with_columns(diff=pl.col("ours").cast(pl.Float64) - pl.col("gt").cast(pl.Float64))
    if edges is not None:
        if len(set(edges)) < len(edges):  # e.g. tied quantiles: merge the bins, drop the names
            edges, labels = sorted(set(edges)), None
        d = d.with_columns(pl.col(column).cut(edges, labels=labels).alias("_group"))
    else:
        d = d.with_columns(pl.col(column).alias("_group"))
    return (d.group_by("_group").agg(
        frames=pl.len(), gt_mean=pl.col("gt").mean().round(2), ours_mean=pl.col("ours").mean().round(2),
        bias=pl.col("diff").mean().round(2), mae=pl.col("diff").abs().mean().round(2))
        .with_columns(share=(pl.col("frames") / pl.col("frames").sum()).round(3)).sort("_group")
        .rename({"_group": column}))
