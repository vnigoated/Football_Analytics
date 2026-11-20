"""Plotting helpers."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def plot_pressing_timeline(
    summary: pl.DataFrame,
    team_names: dict[str, str],
    path: str | Path,
    bin_s: float | None = None,
) -> None:
    """Average pressure on the ball carrier per time bin, one line per pressing team.

    ``bin_s`` defaults to 5 minutes for full matches and 10 seconds for short clips.
    Only frames where a team is out of possession count towards its average.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = summary.drop_nulls("pressing_team_id").with_columns(
        pl.col("pressure_on_carrier").fill_nan(None)
    )
    if bin_s is None:
        bin_s = 300.0 if df["timestamp_s"].max() > 600 else 10.0

    binned = (
        df.with_columns(((pl.col("timestamp_s") // bin_s) * bin_s + bin_s / 2).alias("t_mid"))
        .group_by("period_id", "pressing_team_id", "t_mid")
        .agg(pl.col("pressure_on_carrier").mean().alias("pressure"), pl.len().alias("frames"))
        .sort("period_id", "pressing_team_id", "t_mid")
    )

    periods = binned["period_id"].unique().sort().to_list()
    fig, axes = plt.subplots(1, len(periods), figsize=(12, 3.8), sharey=True, squeeze=False)
    unit, scale = ("min", 60.0) if bin_s >= 60 else ("s", 1.0)
    for ax, period in zip(axes[0], periods):
        part = binned.filter(pl.col("period_id") == period)
        for team_id in part["pressing_team_id"].unique().sort():
            t = part.filter(pl.col("pressing_team_id") == team_id)
            ax.plot(
                t["t_mid"] / scale,
                t["pressure"],
                marker="o",
                label=team_names.get(team_id, team_id),
            )
        ax.set_title(f"Period {period}")
        ax.set_xlabel(f"Match clock ({unit}), {bin_s / scale:g}-{unit} bins")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Mean pressure on ball carrier")
    axes[0][0].legend(title="Pressing team", fontsize=8, loc="best")
    fig.suptitle("Pressing intensity: full IDSSE tracking (ground-truth baseline)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
