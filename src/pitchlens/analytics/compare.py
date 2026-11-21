"""Compare pressing on simulated broadcast tracking against full tracking (RQ1)."""

from __future__ import annotations

import polars as pl

KEYS = ["period_id", "frame_id"]
METRICS = ["pressure_on_carrier", "pressure_total", "n_active_pressers"]


def join_frames(gt: pl.DataFrame, sim: pl.DataFrame) -> pl.DataFrame:
    """Per-frame ground truth next to simulated values (suffix ``_sim``).

    The pressing team always comes from the ground truth. If the camera shows no
    defender at all, the simulated pressure is 0 (nothing to measure on screen).
    """
    sim = sim.select(*KEYS, *METRICS).with_columns(
        pl.col("pressure_on_carrier").fill_nan(None)
    )
    out = gt.with_columns(pl.col("pressure_on_carrier").fill_nan(None)).join(
        sim, on=KEYS, how="left", suffix="_sim"
    )
    gt_ok = pl.col("pressure_on_carrier").is_not_null()
    return out.drop_nulls("pressing_team_id").with_columns(
        *[
            pl.when(gt_ok).then(pl.col(f"{m}_sim").fill_null(0)).otherwise(pl.col(f"{m}_sim"))
            .alias(f"{m}_sim")
            for m in METRICS
        ]
    )


def bias_table(joined: pl.DataFrame, high_pressure: float = 0.5) -> pl.DataFrame:
    """One row per match, pressing team (and ``view_length`` if present)."""
    cols = ("camera", "view_length", "match_id", "pressing_team_name")
    groups = [c for c in cols if c in joined.columns]
    p, ps = pl.col("pressure_on_carrier"), pl.col("pressure_on_carrier_sim")
    valid = joined.drop_nulls("pressure_on_carrier")
    return (
        valid.group_by(groups)
        .agg(
            p.mean().alias("mean_pressure_gt"),
            ps.mean().alias("mean_pressure_sim"),
            (ps - p).abs().mean().alias("frame_mae"),
            pl.corr(p, ps).alias("frame_corr"),
            ((p > high_pressure).mean() * 100).alias("pct_high_gt"),
            ((ps > high_pressure).mean() * 100).alias("pct_high_sim"),
            pl.col("n_active_pressers").mean().alias("active_gt"),
            pl.col("n_active_pressers_sim").mean().alias("active_sim"),
            pl.col("pressure_total").fill_nan(None).mean().alias("total_gt"),
            pl.col("pressure_total_sim").fill_nan(None).mean().alias("total_sim"),
        )
        .with_columns(
            ((pl.col("mean_pressure_sim") / pl.col("mean_pressure_gt") - 1) * 100).alias(
                "rel_bias_pressure_pct"
            ),
            ((pl.col("total_sim") / pl.col("total_gt") - 1) * 100).alias("rel_bias_total_pct"),
            ((pl.col("active_sim") / pl.col("active_gt") - 1) * 100).alias("rel_bias_active_pct"),
        )
        .sort(groups)
    )
