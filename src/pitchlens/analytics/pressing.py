"""Pressing intensity (Bekkers, 2025) via the ``unravelsports`` package.

``unravelsports`` returns, per frame, a matrix of probabilities that each defender
can intercept each attacker (ball merged into the carrier). We keep the raw matrices
and add simple per-frame summaries that are easy to plot and compare between
full tracking and broadcast-like tracking (RQ1).
"""

from __future__ import annotations

import numpy as np
import polars as pl


def compute_pressing_intensity(
    tracking_dataset,
    speed_threshold: float = 2.0,
    **fit_kwargs,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run unravelsports PressingIntensity on a kloppy TrackingDataset.

    Returns:
        (pti_output, tracking): raw per-frame matrices and the smoothed long-format
        tracking frame.

    Note: in unravelsports 1.2.1 the docstring for ``orient`` is reversed. In the code,
    ``"ball_owning"`` gives rows = defenders, columns = attackers. ``summarise_frames``
    detects the orientation from the ball carrier, so it works either way.
    """
    from unravel.soccer import KloppyPolarsDataset, PressingIntensity

    kp = KloppyPolarsDataset(kloppy_dataset=tracking_dataset)
    kp.data, n_dropped = drop_frames_without_carrier(kp.data)
    if n_dropped:
        print(f"Dropped {n_dropped} frames with no ball carrier in the possessing team")
    model = PressingIntensity(dataset=kp)
    model.fit(
        method="teams",
        ball_method="max",
        orient="ball_owning",
        speed_threshold=speed_threshold,
        **fit_kwargs,
    )
    return model.output, kp.data


def drop_frames_without_carrier(tracking: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Remove frames where no player of the ball-owning team is flagged as ball carrier.

    unravelsports (``ball_method="max"``) assumes exactly such a carrier exists and
    crashes otherwise. On IDSSE match J03WPY this affects 48 of 81,460 frames (0.06%),
    e.g. loose balls.
    """
    keys = ["period_id", "frame_id"]
    valid = (
        tracking.group_by(keys)
        .agg(
            (pl.col("is_ball_carrier") & (pl.col("team_id") == pl.col("ball_owning_team_id")))
            .any()
            .alias("_ok")
        )
        .filter(pl.col("_ok"))
        .select(keys)
    )
    n_before = tracking.select(keys).unique().height
    out = tracking.join(valid, on=keys, how="semi")
    return out, n_before - valid.height


def summarise_matches(
    frames: pl.DataFrame,
    fps: float = 25.0,
    high_pressure: float = 0.5,
) -> pl.DataFrame:
    """One row per match and pressing team from concatenated ``summarise_frames`` outputs.

    Expects ``match_id`` and ``pressing_team_name`` columns (added by the CLI).
    """
    p = pl.col("pressure_on_carrier").fill_nan(None)
    return (
        frames.drop_nulls("pressing_team_id")
        .group_by("match_id", "pressing_team_name")
        .agg(
            (pl.len() / fps / 60).round(1).alias("minutes_out_of_possession"),
            p.mean().round(3).alias("mean_pressure"),
            p.median().round(3).alias("median_pressure"),
            ((p > high_pressure).mean() * 100).round(1).alias("pct_high_pressure"),
            pl.col("n_active_pressers").mean().round(2).alias("mean_active_pressers"),
        )
        .sort("match_id", "mean_pressure", descending=[False, True])
    )


def summarise_frames(
    pti_output: pl.DataFrame,
    tracking: pl.DataFrame,
    active_threshold: float = 0.5,
) -> pl.DataFrame:
    """Reduce per-frame probability matrices to per-frame pressing summaries.

    Columns of the result:
        pressing_team_id     team without the ball (rows of the matrix)
        pressure_on_carrier  P(at least one defender can intercept the ball carrier)
        pressure_total       sum of all defender-attacker probabilities
        n_active_pressers    defenders with max probability > ``active_threshold``
    """
    team_of = dict(
        tracking.select("id", "team_id").unique().iter_rows()
    )
    carriers = (
        tracking.filter(pl.col("is_ball_carrier"))
        .select("period_id", "frame_id", pl.col("id").alias("carrier_id"))
        .unique(subset=["period_id", "frame_id"], keep="first")
    )
    joined = pti_output.join(carriers, on=["period_id", "frame_id"], how="left")

    records = []
    for row in joined.iter_rows(named=True):
        P = np.asarray(row["probability_to_intercept"], dtype=np.float64)
        rows, cols = list(row["rows"]), list(row["columns"])
        carrier = row.get("carrier_id")
        # Normalise orientation: rows must be defenders, columns attackers.
        if carrier is not None and carrier in rows and carrier not in cols:
            P, rows, cols = P.T, cols, rows
        rec = {
            "period_id": row["period_id"],
            "frame_id": row["frame_id"],
            "timestamp_s": row["timestamp"].total_seconds(),
            "pressing_team_id": team_of.get(rows[0]) if rows else None,
            "pressure_on_carrier": np.nan,
            "pressure_total": np.nan,
            "n_active_pressers": 0,
        }
        if P.ndim == 2 and P.size:
            rec["pressure_total"] = float(np.nansum(P))
            rec["n_active_pressers"] = int((np.nanmax(P, axis=1) > active_threshold).sum())
            if carrier in cols:
                p = np.nan_to_num(P[:, cols.index(carrier)], nan=0.0)
                rec["pressure_on_carrier"] = float(1.0 - np.prod(1.0 - p))
        records.append(rec)
    return pl.DataFrame(records)
