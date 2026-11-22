"""Confidence intervals for the visibility bias (RQ2, first step).

Frames 40 ms apart are almost identical, so resampling single frames would give
far too narrow intervals. We use a **block bootstrap**: each match period is cut
into blocks of ``block_s`` seconds and whole blocks are resampled.

- Per team-match: resample blocks of that team's out-of-possession frames.
- Overall (mean over team-matches): a two-level bootstrap. Resample *matches*
  (both teams of a match stay together), then use each team's block-bootstrap
  replicate, so the interval reflects both match-to-match and within-match variation.

The bias is ``sum(sim) / sum(gt) - 1``, i.e. the same ratio of means reported by
``bias_table``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

GROUP = ["camera", "view_length", "match_id", "pressing_team_name"]
TARGETS = {
    "pressure": ("pressure_on_carrier", "pressure_on_carrier_sim"),
    "active": ("n_active_pressers", "n_active_pressers_sim"),
}


def block_sums(joined: pl.DataFrame, block_s: float = 60.0) -> pl.DataFrame:
    """Per (group, block): sums of ground-truth and simulated values and frame count."""
    groups = [c for c in GROUP if c in joined.columns]
    aggs = [pl.len().alias("n")]
    for name, (gt, sim) in TARGETS.items():
        aggs += [
            pl.col(gt).cast(pl.Float64).sum().alias(f"{name}_gt"),
            pl.col(sim).cast(pl.Float64).sum().alias(f"{name}_sim"),
        ]
    return (
        joined.drop_nulls("pressure_on_carrier")
        .with_columns((pl.col("timestamp_s") // block_s).cast(pl.Int64).alias("block"))
        .group_by(*groups, "period_id", "block")
        .agg(aggs)
        .sort(*groups, "period_id", "block")
    )


def _boot_ratio(gt: np.ndarray, sim: np.ndarray, n_boot: int, rng: np.random.Generator):
    """Bootstrap replicates of sum(sim)/sum(gt) - 1 over blocks (in %)."""
    k = len(gt)
    counts = rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (counts @ sim / (counts @ gt) - 1.0) * 100.0


def bootstrap_bias(
    joined: pl.DataFrame,
    block_s: float = 60.0,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Point estimates and percentile CIs of the relative bias (%).

    Returns ``(per_team, overall)``. Columns ``<target>_bias``, ``<target>_lo``,
    ``<target>_hi`` for targets ``pressure`` (on the ball carrier) and ``active``
    (number of active pressers).
    """
    rng = np.random.default_rng(seed)
    lo_q, hi_q = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    blocks = block_sums(joined, block_s)
    groups = [c for c in GROUP if c in blocks.columns]
    cfg_cols = [c for c in ("camera", "view_length") if c in groups]

    per_team, overall = [], []
    for cfg, part in blocks.group_by(cfg_cols, maintain_order=True) if cfg_cols else [((), blocks)]:
        reps: dict[str, dict[tuple, np.ndarray]] = {t: {} for t in TARGETS}
        team_keys = part.select(groups).unique(maintain_order=True).rows()
        for key in team_keys:
            b = part.filter(pl.all_horizontal([pl.col(c) == v for c, v in zip(groups, key)]))
            row = dict(zip(groups, key))
            row["n_blocks"] = b.height
            for t in TARGETS:
                gt, sim = b[f"{t}_gt"].to_numpy(), b[f"{t}_sim"].to_numpy()
                r = _boot_ratio(gt, sim, n_boot, rng)
                reps[t][key] = r
                row[f"{t}_bias"] = (sim.sum() / gt.sum() - 1.0) * 100.0
                row[f"{t}_lo"], row[f"{t}_hi"] = np.nanpercentile(r, [lo_q, hi_q])
            per_team.append(row)

        # two-level bootstrap over matches
        match_pos = groups.index("match_id") if "match_id" in groups else None
        matches = sorted({k[match_pos] for k in team_keys}) if match_pos is not None else [None]
        teams_of = {m: [k for k in team_keys if match_pos is None or k[match_pos] == m] for m in matches}
        row = dict(zip(cfg_cols, cfg if isinstance(cfg, tuple) else (cfg,)))
        row["team_matches"] = len(team_keys)
        row["matches"] = len(matches)
        picks = rng.integers(0, len(matches), size=(n_boot, len(matches)))
        for t in TARGETS:
            point = np.mean([r[f"{t}_bias"] for r in per_team[-len(team_keys):]])
            mat = {k: v for k, v in reps[t].items()}
            boot = np.empty(n_boot)
            for i in range(n_boot):
                vals = [mat[k][i] for j in picks[i] for k in teams_of[matches[j]]]
                boot[i] = np.nanmean(vals)
            row[f"{t}_bias"] = point
            row[f"{t}_lo"], row[f"{t}_hi"] = np.nanpercentile(boot, [lo_q, hi_q])
        overall.append(row)

    return pl.DataFrame(per_team), pl.DataFrame(overall)
