# Experiment 2: replication on Metrica (second data source)

Same experiment as [1](exp1_visibility_bias.md) / [1b](exp1b_realistic_camera.md) / [RQ2](rq2_confidence_intervals.md),
on the 2 Metrica Sports open sample games (anonymised teams, 25 fps, all 22 players + ball).
Metrica's tracking has no possession or ball-in-play flag, so both are **rebuilt from Metrica's event file**
(`pitchlens.data.metrica`; owner = team of the latest pass / recovery / shot / set piece / ball lost; dead ball
from ball out, foul, or shot out/goal until the restart). Run on Colab A100 (cell 4c, 392 s).

## Ground truth (full tracking)

| Game | Team | Min. out of possession | Mean pressure | % high pressure | Active pressers |
|---|---|---|---|---|---|
| 1 | Away | 29.2 | 0.262 | 27.2 | 1.00 |
| 1 | Home | 26.6 | 0.256 | 26.1 | 1.00 |
| 2 | Away | 28.2 | 0.268 | 27.7 | 0.94 |
| 2 | Home | 27.2 | 0.261 | 27.3 | 0.91 |

Similar to IDSSE (mean pressure 0.22–0.36), with slightly fewer active pressers (IDSSE 0.99–1.45).

## Visibility bias (4 team-matches, 95 % block-bootstrap CIs)

| Camera | Players visible | Pressure on ball carrier | Active pressers | IDSSE, same camera |
|---|---|---|---|---|
| Simple, 40 m | 15.7 | −0.17 % [−0.24, −0.11] | −16.0 % [−19.0, −13.1] | −20.3 % at 14.5 players |
| Realistic (lag 0.5 s, persp. 0.3, zoom 0.5), 32 m | 15.5 | −3.3 % [−3.9, −2.8] | −18.6 % [−21.3, −16.3] | −22.8 % at 14.5 players |

## Interpretation

- **The same camera shows about one more player on Metrica** (15.5 vs 14.5): the teams are more compact. Because
  each extra visible player reduces the undercount by about 3 percentage points on IDSSE (−20 % at 14.5 players,
  −6 % at 19.3), the IDSSE results adjusted to 15.5 visible players are about −17 % (simple) and −20 % (realistic),
  in line with Metrica's −16 % and −18.6 %.
- **Pressure on the ball carrier** shows the same lag effect: −3.3 % (Metrica) vs −3.8 % (IDSSE).
- **New insight:** the size of the bias depends on team compactness (how many players fit in the camera view),
  so a single correction factor across leagues or styles would be wrong.

## Equal-visibility comparison (Colab cell 4e)

The camera was narrowed on Metrica until it showed the same number of players as on IDSSE (about 14.5):

| Camera | Dataset | View | Players visible | Pressure on ball carrier | Active pressers |
|---|---|---|---|---|---|
| Realistic | IDSSE (7 games) | 32 m | 14.45 | −3.8 % [−4.3, −3.4] | −22.8 % [−25.1, −20.5] |
| Realistic | **Metrica (2 games)** | **29 m** | **14.48** | **−4.0 % [−4.7, −3.4]** | **−22.3 % [−25.3, −19.6]** |
| Simple | IDSSE | 40 m | 14.55 | −0.12 % [−0.15, −0.09] | −20.3 % [−22.7, −18.0] |
| Simple | **Metrica** | **36 m** | **14.60** | **−0.35 % [−0.47, −0.26]** | **−19.4 % [−22.7, −16.2]** |

**At equal visibility the two data sources agree within half a percentage point** for the realistic camera,
and the confidence intervals overlap almost completely. Across the Metrica sweep (13.8–15.7 visible players),
each extra visible player reduces the undercount of active pressers by about 3.5 percentage points
(about 3 on IDSSE).

**Conclusion:** the bias is driven by **how many players the camera shows**, not by the league or the data
provider. The number of visible players, which can be measured on real broadcast video, is therefore the key
quantity for correcting broadcast pressing metrics.

## Caveats

- Only 2 games: the match-level bootstrap has 2 clusters, so the intervals are likely too narrow. Treat Metrica as a
  qualitative replication.
- Possession is rebuilt from hand-annotated events; IDSSE possession comes from the provider.
