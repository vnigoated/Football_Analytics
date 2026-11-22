"""Correct broadcast pressing metrics for visibility bias (RQ3, first version).

A "broadcast" is one stadium camera filming one match. Across broadcasts, the relative
undercount of active pressers is close to linear in the mean number of visible players.
We fit ``bias = k * visible + b`` and divide the broadcast value by ``1 + bias``.

Evaluation is strict: for each broadcast, the line is fitted without its stadium camera
**and** without its match, so the correction never sees either.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def broadcast_table(joined: pl.DataFrame) -> pl.DataFrame:
    """One row per (camera, match, pressing team): ground truth, broadcast value, visibility."""
    valid = joined.drop_nulls("pressure_on_carrier")
    vis = valid.group_by("camera", "match_id").agg(pl.col("players_visible").first())
    return (
        valid.group_by("camera", "match_id", "pressing_team_name")
        .agg(
            pl.col("n_active_pressers").mean().alias("gt"),
            pl.col("n_active_pressers_sim").mean().alias("sim"),
        )
        .join(vis, on=["camera", "match_id"])
        .with_columns(((pl.col("sim") / pl.col("gt") - 1) * 100).alias("bias"))
    )


def evaluate_correction(table: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, float]]:
    """Leave-one-camera-and-match-out evaluation. Returns per-row errors and a summary."""
    out = []
    for (cam, match), test in table.group_by("camera", "match_id"):
        train = table.filter((pl.col("camera") != cam) & (pl.col("match_id") != match))
        k, b = np.polyfit(train["players_visible"], train["bias"], 1)
        pred = k * test["players_visible"].to_numpy() + b
        corrected = test["sim"].to_numpy() / (1 + pred / 100)
        out.append(
            test.with_columns(
                pl.Series("predicted_bias", pred),
                pl.Series("error_corrected", (corrected / test["gt"].to_numpy() - 1) * 100),
            )
        )
    res = pl.concat(out)
    raw, cor = res["bias"].to_numpy(), res["error_corrected"].to_numpy()
    k, b = np.polyfit(table["players_visible"], table["bias"], 1)
    summary = {
        "broadcasts": float(res.height),
        "slope_pp_per_player": float(k),
        "intercept": float(b),
        "raw_mean_error": float(raw.mean()),
        "raw_mean_abs_error": float(np.abs(raw).mean()),
        "corrected_mean_error": float(cor.mean()),
        "corrected_mean_abs_error": float(np.abs(cor).mean()),
        "corrected_p95_abs_error": float(np.percentile(np.abs(cor), 95)),
    }
    return res, summary
