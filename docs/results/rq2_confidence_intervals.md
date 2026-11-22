# RQ2 (step 1): confidence intervals for the visibility bias

How sure are we about the numbers in [Experiment 1](exp1_visibility_bias.md) and [1b](exp1b_realistic_camera.md)?

**Method.** Two-level block bootstrap (2,000 replicates, 95 % percentile intervals):
1. within each team-match, the out-of-possession frames are cut into **60-second blocks** and whole blocks are resampled
   (frames 40 ms apart are nearly identical, so resampling single frames would give intervals that are far too narrow);
2. for the overall number, **matches** are resampled (both teams of a match together) on top of that,
   so the interval covers both match-to-match and within-match variation.

Reproduce: `pitchlens visibility-ci` (reads the saved `vis_*.parquet`, about 15 s on a laptop).

## Results (7 IDSSE matches, 14 team-matches)

| Camera | Players visible | Pressure on ball carrier | Active pressers |
|---|---|---|---|
| Simple, 40 m | 14.5 | −0.12 % [−0.15, −0.09] | −20.3 % [−22.7, −18.0] |
| Perspective 0.3, 40 m | 14.4 | −0.65 % [−0.90, −0.44] | −20.9 % [−23.3, −18.7] |
| Lag 0.5 s, 40 m | 14.3 | −4.1 % [−4.6, −3.7] | −24.1 % [−26.5, −21.9] |
| Lag 1.0 s, 40 m | 14.0 | −7.9 % [−8.9, −7.0] | −27.5 % [−29.9, −25.3] |
| **Realistic, calibrated, 32 m** | **14.5** | **−3.8 % [−4.3, −3.4]** | **−22.8 % [−25.1, −20.5]** |

**Robustness.** Changing the block length to 30 s or 120 s moves the interval limits by at most 0.3 percentage points.
For the calibrated realistic camera, none of the 14 per-team intervals for active pressers contains 0
(widths 5–9 percentage points; see `outputs/visibility_bias/ci_per_team.csv`).

## What it means

- The undercount of active pressers is **clearly not noise**: about −23 %, and even the optimistic end of the interval is −20 %.
- Pressure on the ball carrier is small but **reliably negative** once the camera lags (−3.8 %, interval excludes 0).
- Lag effects are separable: the intervals for 0.5 s and 1.0 s lag do not overlap for pressure on the carrier.

## Limitations

- Intervals cover sampling variation across these 7 matches only. Uncertainty in the camera parameters
  (lag, zoom, perspective) is not included yet: next, sweep or fit them (SkillCorner) and propagate.
- 7 matches from one competition, Düsseldorf in 5: the Metrica replication tests generalisation.
- Differences between cameras are compared by overlap only; a paired bootstrap would be more powerful.
