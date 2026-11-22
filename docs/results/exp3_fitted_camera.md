# Experiment 3: a camera fitted to real broadcast footage

Experiments [1](exp1_visibility_bias.md) and [1b](exp1b_realistic_camera.md) used a rectangular camera with **assumed**
lag, zoom and perspective. Here the camera is **measured** from real broadcasts and then reused on full tracking data.

## Data

[SkillCorner open data](https://github.com/SkillCorner/opendata) (MIT licence, credit SkillCorner): 20 A-League 2024/25
matches of broadcast tracking at 10 fps. Every frame gives the **camera footprint on the pitch** (4 corners) and, per
player, whether they were **on screen** (`is_detected`) or extrapolated.

## Camera model (`WedgeCamera`)

The main camera is a fixed point behind the near touchline. A player is visible if their bearing from the camera is
within `pan ± fov/2` and they are beyond the bottom edge of the image.

| Part | How it is measured | Fitted result (10 matches, 370k frames) |
|---|---|---|
| Position | intersection of the image's left and right edges | on the halfway line, 57–96 m from the centre depending on the stadium (median 64 m) |
| Pan | least squares on ball bearing and players' mean bearing | follows ball (0.57) **and** players' centre (0.56) equally; R² = 0.97 |
| Lag | grid search 0–1 s | **0 s** (error grows with every 0.1 s of assumed lag) |
| Zoom (field of view) | on ball distance and player spread | 27° median (19–40°); zooms in for far play, out when players spread; R² = 0.48 |
| Tilt (bottom edge) | on ball distance | moves away with the ball; R² = 0.60 |

Parameters: `configs/cameras/wedge_skillcorner.json`. Reproduce: `pitchlens fit-camera`.

## Validation on 10 held-out matches

How well does each camera predict **which players were actually on screen**?

| Camera | Agreement with `is_detected` | Visible players / frame (real: 14.65) |
|---|---|---|
| Real footprint (upper bound) | 96.8 % | – |
| **Fitted WedgeCamera** | **87.1 %** (84.0–88.9 per match) | **14.65** |
| Rectangle, 40 m (Exp. 1) | 82.9 % | 15.76 |
| Rectangle, 32 m + lag 0.5 s + perspective + zoom (Exp. 1b) | 82.2 % | 15.29 |
| WedgeCamera following the ball only | 82.0 % | 13.30 |
| WedgeCamera + random AR(1) framing noise | 80.7 % | 13.66 |

The fitted camera beats the rectangle on all 10 test matches. Framing the players (not only the ball) is what makes the
difference; random framing noise makes predictions worse, so it is not used.

## Pressing bias with the fitted camera (95 % block-bootstrap CIs)

| Data | Players visible | Pressure on ball carrier | Active pressers | Total pressure |
|---|---|---|---|---|
| IDSSE, 7 matches | 13.8 | **−0.54 % [−0.68, −0.43]** | **−21.4 % [−23.8, −19.1]** | −22.2 % |
| Metrica, 2 matches | 14.5 | −0.68 % [−1.00, −0.45] | −17.7 % [−20.8, −15.1] | – |

Per team (IDSSE): −16 % to −27 %, no interval contains 0; team ranking by active pressers keeps Spearman 0.90.

## What changes compared with Experiments 1/1b

1. **The main finding holds with a measured camera:** broadcast tracking misses about **one fifth of a team's active
   pressers** (−21 % IDSSE, −18 % Metrica).
2. **Correction:** Experiment 1b attributed a −3.8 % bias in pressure on the ball carrier to camera lag. Real cameras show
   **no measurable lag**, and with the fitted camera this bias is only about **−0.5 %**. Carrier-centred metrics are
   therefore close to unbiased on broadcast data; team-wide pressing metrics are not.
3. The camera operator frames **the play**, not only the ball. Simulations that only follow the ball under-estimate
   visibility (13.3 vs 14.65 players).

## Limitations

- One typical camera position is used for IDSSE/Metrica; real stadiums differ (57–96 m), and so does visibility per
  match (13.5–16.2 players). Propagating stadium-to-stadium variation is the next step (RQ2 part 2).
- `is_detected` also reflects SkillCorner's detector (occlusions, replays), which caps agreement at about 97 %.
- Undetected SkillCorner positions are extrapolated, so the players' centre used for framing is approximate.
