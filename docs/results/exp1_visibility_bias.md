# Experiment 1: does a TV camera distort pressing metrics? (first run)

**Setup.** Full IDSSE optical tracking (7 Bundesliga / 2. Bundesliga matches, 25 fps) is the ground truth.
A virtual broadcast camera (rectangle, full pitch width, centred on the ball with no lag, clipped to the pitch; edges on a pitch line stay open)
hides every player outside its view. Pressing intensity (Bekkers 2025, `unravelsports`, speed threshold 2 m/s) is recomputed on what the camera sees
and compared frame by frame with the ground truth. The ball carrier is always visible.

Reproduce: `pitchlens pressing-baseline --match <id>` → `pitchlens visibility-bias --match <id> --view-lengths 30 40 60` → `pitchlens visibility-summary`.

**Sanity check.** With a full-pitch camera (105 m) the result is identical to the ground truth on all 81,412 frames of J03WPY.

## Results (14 team-matches)

| Camera view length | Players visible / frame | Bias: pressure on ball carrier | Bias: active pressers | Bias: total pressure | Team ranking (Spearman, active pressers) |
|---|---|---|---|---|---|
| 30 m | 11.8 | −0.8 % | **−30 %** (range −35 to −25) | −31 % | 0.87 |
| 40 m (≈ real broadcast, ~14 players) | 14.5 | −0.1 % | **−20 %** (range −25 to −15) | −21 % | 0.90 |
| 60 m | 19.3 | 0.0 % | −6 % (range −10 to −3) | −6 % | 0.94 |
| 105 m (full pitch) | 22.0 | 0 | 0 | 0 | 1.00 |

## What it means

1. **Pressure on the ball carrier is robust.** The defenders who can reach the ball are close to it, and the camera follows the ball, so they are almost always on screen.
2. **Team-wide pressing counts are not.** "How many players are pressing" and "total pressure" are undercounted by about **20 %** at realistic broadcast zoom, because pressers covering passing lanes far from the ball are off screen.
3. **The bias differs between teams** (−15 % to −25 % at 40 m), so it is not a constant you can simply correct for, and it reshuffles team rankings a little.

## Limitations (next steps)

- Best-case camera: no lag, no zoom changes, rectangular footprint. **Follow-up:** [Experiment 1b](exp1b_realistic_camera.md) adds lag, perspective and zoom.
- Only one dataset/league and Düsseldorf appears in 5 of 7 matches. Next: repeat on Metrica (complete, 22/22 players verified).
- No uncertainty intervals yet (bootstrap over possessions → RQ2).
