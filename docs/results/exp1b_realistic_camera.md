# Experiment 1b: a more realistic virtual camera

> **Update (Experiment 3):** a camera fitted to real broadcasts shows **no measurable lag**. The lag values here were
> assumptions; with the fitted camera the bias in pressure on the ball carrier is about −0.5 %, not −3.8 %.
> The undercount of active pressers (about −21 %) is confirmed. See [Experiment 3](exp3_fitted_camera.md).

Follow-up to [Experiment 1](exp1_visibility_bias.md). Same data (7 IDSSE matches, full tracking as ground truth), same pressing model,
but the camera now has three optional effects (`pitchlens.simulation.camera.BroadcastCamera`):

| Effect | Parameter | Meaning |
|---|---|---|
| Lag | `lag_s` | camera centre follows the ball with an exponential delay (time constant in seconds) |
| Perspective | `perspective` | main camera on the near touchline: footprint 30 % shorter on the near side, 30 % longer on the far side (same average) |
| Zoom | `zoom_gain` | view widens up to 50 % when the ball moves fast (smoothed speed ≥ 15 m/s) |

Reproduce, e.g.: `pitchlens visibility-bias --match J03WPY --view-lengths 32 --lag-s 0.5 --perspective 0.3 --zoom-gain 0.5 --tag realistic_cal`, then `pitchlens visibility-summary`.

## Results (14 team-matches, relative bias vs. full tracking)

| Camera | View length | Players visible | Pressure on ball carrier | Active pressers | Total pressure |
|---|---|---|---|---|---|
| Simple (Exp. 1) | 40 m | 14.5 | −0.1 % | −20 % | −21 % |
| Perspective 0.3 | 40 m | 14.4 | −0.6 % | −21 % | −21 % |
| Lag 0.5 s | 40 m | 14.3 | −4.1 % | −24 % | −24 % |
| Lag 1.0 s | 40 m | 14.0 | −7.9 % | −28 % | −28 % |
| Zoom 0.5 | 40 m | 16.8 | 0.0 % | −12 % | −13 % |
| All three (uncalibrated) | 40 m | 16.8 | −2.8 % | −14 % | −14 % |
| **All three, calibrated to ~14.5 players** | **32 m** | **14.5** | **−3.8 %** (worst team −5.6 %) | **−23 %** (range −27 to −18 %) | **−23 %** |

Team ranking by active pressers (Spearman, ground truth vs. camera): 0.90 for both the simple and the calibrated realistic camera.

## What it means

1. **Camera lag is what hurts most.** When the camera trails the ball, defenders closing down the carrier are cut off. Even the
   carrier-centred metric, which looked safe in Experiment 1 (−0.1 %), is then undercounted by 4–8 %.
2. **Perspective barely matters** at this zoom level.
3. **Zoom must be compared at equal visibility.** Zooming out shows more players and looks better (−12 %), but at the same
   average visibility as real TV (~14.5 players) the realistic camera is slightly *worse* than the simple one (−23 % vs −20 %),
   because the lag effect dominates.
4. **Bottom line:** at realistic broadcast visibility, team pressing counts are undercounted by about a fifth to a quarter,
   pressure on the ball carrier by about 4 %, and the size of the error differs between teams.

## Limitations

- The lag, perspective and zoom values are assumptions, not yet fitted to real broadcast visibility (SkillCorner, next step).
- One league, Düsseldorf in 5 of 7 matches; Metrica replication pending.
- No uncertainty intervals yet (RQ2).
