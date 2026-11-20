"""Plotting helpers."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def plot_pressing_timeline(
    summary: pl.DataFrame,
    team_names: dict[str, str],
    path: str | Path,
    window_s: int = 10,
) -> None:
    """Rolling (time-window) pressure on the ball carrier, one line per pressing team.

    Lines break while a team is in possession, so no value is carried across those gaps.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = (
        summary.drop_nulls("pressing_team_id")
        .sort("period_id", "timestamp_s")
        .with_columns(
            pl.from_epoch((pl.col("timestamp_s") * 1000).cast(pl.Int64), time_unit="ms").alias(
                "_t"
            ),
            pl.col("pressure_on_carrier").fill_nan(None),
        )
    )
    fig, axes = plt.subplots(
        1, df["period_id"].n_unique(), figsize=(12, 3.5), sharey=True, squeeze=False
    )
    for ax, (period,) in zip(axes[0], df.select("period_id").unique().sort("period_id").rows()):
        part = df.filter(pl.col("period_id") == period)
        for team_id in part["pressing_team_id"].unique().sort():
            pressing = pl.col("pressing_team_id") == team_id
            smooth = part.select(
                pl.when(pressing)
                .then(
                    pl.when(pressing)
                    .then(pl.col("pressure_on_carrier"))
                    .rolling_mean_by("_t", window_size=f"{window_s}s")
                )
                .otherwise(None)
                # break the line after dead-ball gaps (> 1 s jump in match clock)
                .pipe(lambda e: pl.when(pl.col("timestamp_s").diff() > 1.0).then(None).otherwise(e))
            ).to_series()
            ax.plot(
                part["timestamp_s"] / 60,
                smooth.to_numpy(),
                label=team_names.get(team_id, team_id),
            )
        ax.set_title(f"Period {period}")
        ax.set_xlabel("Minute (ball in play)")
    axes[0][0].set_ylabel("Pressure on ball carrier")
    axes[0][-1].legend(title="Pressing team", fontsize=8)
    fig.suptitle("Pressing intensity: full IDSSE tracking (ground-truth baseline)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
