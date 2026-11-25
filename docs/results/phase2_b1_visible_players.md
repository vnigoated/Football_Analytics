# Phase 2 · B1: counting visible players on real video

**Notebook:** `colab/08_visible_players.ipynb` · **Code:** `pitchlens.analytics.visible_count` · **Run 1:** pending

## Question

The Phase 1 correction needs one input per broadcast: the mean number of visible players per frame (both teams,
goalkeepers included, no referees). Its slope is about 4.1 percentage points of active-presser bias per player.
How accurately does our pipeline measure that number on real broadcast video?

## Method

**Data.** Every labelled frame of SN-GSR valid and test. The labels count the labelled players and goalkeepers in
each frame, which is A1's definition: 14.84 per frame on valid.

**Two pipeline counts:**
- **tracks:** every tracked person whose role is player or goalkeeper. No homography is needed.
- **game state:** only those that also received a pitch position, as in A6.

**Calibration.** A linear fit `labels ≈ a · ours + b` on the tuning half of valid only. It is reported both with and
without the fit, on the held-out half and on test.

**Error measures:**
- the mean error (bias), with a 95% interval from resampling whole clips;
- the frame-level mean absolute error;
- the clip-level error of each 30-second clip mean.

**Error sources.** Errors are broken down by zoom (median labelled player height in pixels), by how many players the
labels show, and by the homography state of the frame (own / filled / held / none).

**Translation to pressing.** An error of δ players in a broadcast's mean shifts the predicted bias by 4.1 · δ
percentage points.

## Run 1 (2026-11-25, Colab; 80,250 labelled frames, 57 s)

### Accuracy (players + goalkeepers per frame)

The 95% intervals come from resampling whole clips. "Implied shift" is 4.1 × bias, in percentage points (pp) of
active-presser bias.

| Count | Set | Calibrated | Labels | Ours | Bias [95% CI] | Frame MAE | Clip MAE | Clip p95 abs. | Clip r | Implied shift |
|---|---|---|---|---|---|---|---|---|---|---|
| **tracks** | held out (29) | no | 14.94 | 13.89 | −1.05 [−1.41, −0.75] | 1.15 | 1.07 | 2.50 | 0.945 | −4.3 pp |
| **tracks** | held out (29) | **yes** | 14.94 | 14.62 | **−0.32 [−0.68, −0.01]** | 1.10 | 0.67 | 1.90 | 0.945 | **−1.3 pp** |
| **tracks** | test (49) | no | 13.17 | 12.45 | −0.72 [−0.95, −0.51] | 0.94 | 0.82 | 2.33 | 0.956 | −3.0 pp |
| **tracks** | test (49) | **yes** | 13.17 | 13.26 | **+0.09 [−0.13, 0.29]** | 1.09 | 0.58 | 1.71 | 0.956 | **+0.4 pp** |
| game state | held out | no | 14.94 | 13.52 | −1.42 [−1.85, −1.02] | 1.51 | 1.45 | 3.62 | 0.907 | −5.8 pp |
| game state | held out | yes | 14.94 | 15.00 | +0.06 [−0.50, 0.61] | 1.93 | 1.33 | 2.73 | 0.907 | +0.3 pp |
| game state | test | no | 13.17 | 10.86 | −2.31 [−3.07, −1.60] | 2.52 | 2.39 | 7.73 | 0.763 | −9.5 pp |
| game state | test | yes | 13.17 | 13.62 | +0.45 [−0.06, 0.92] | 2.45 | 1.38 | 3.69 | 0.763 | +1.8 pp |

The calibrations were fitted on the tuning half (labels = a · ours + b): tracks a = 0.943, b = 1.52; game state
a = 0.52, b = 7.97.

### Where the errors come from (held-out half + test, tracks count, raw)

**By zoom** (median labelled player height; quartile edges at 91, 106 and 124 px):

| Zoom | Labels | Ours | Bias |
|---|---|---|---|
| close | 11.03 | 10.45 | −0.58 |
| mid-close | 13.07 | 12.42 | −0.65 |
| mid-wide | 14.59 | 13.80 | −0.78 |
| wide | 16.57 | 15.22 | −1.35 |

**By crowding** (players in the labels):

| Players | Bias |
|---|---|
| ≤ 10 | −0.38 |
| 11–14 | −0.62 |
| 15–18 | −0.97 |
| ≥ 19 | −1.75 |

**By homography state:**

| State | Share of frames | Bias, tracks | Bias, game state |
|---|---|---|---|
| own | 74.8% | −0.90 | −0.91 |
| filled | 12.2% | −0.77 | −0.77 |
| held | 3.2% | −0.64 | −0.65 |
| none | 9.8% | −0.53 | **−12.18** (no positions at all) |

## Findings

1. **The pipeline undercounts visible players by 0.7–1.1 per frame (5–7%).** The miss grows with wide shots and
   crowds, up to −1.75 with 19 or more players: small players and players hidden in groups are the ones missed. Clip
   means still follow the labels closely (r = 0.95–0.96).
2. **A linear calibration fitted on separate clips removes the bias.** The residual is −0.32 [−0.68, −0.01] players
   on the held-out half and +0.09 [−0.13, 0.29] on test, which shifts the corrected pressing value by −1.3 and
   +0.4 pp. That is below the correction's own error (3.1% mean absolute error, paper v1).
3. **Count from the tracks, not from the game state.** Frames without a homography (9.8%; close-ups and views
   without enough pitch landmarks) have no pitch positions, which drives the game-state count far off on test
   (−2.31). The tracks count does not need a homography, and those frames are fine for it (−0.53).
4. **A single 30-second clip is noisier:** the 95th percentile of clip errors is 1.7–1.9 players after calibration
   (7–8 pp). A broadcast is a whole match, about 180 such clips, so this noise mostly averages out; the systematic
   part is what the calibration removes. B2 propagates both properly.
5. **Definition check:** the Phase 1 "visible players" counts players inside the camera footprint. The labels count
   annotated players, close to the same thing: A1 found 14.84 labelled against 14.65 for the fitted camera.

**Figure:** Drive `outputs/b1_visible/b1_visible_players.{png,pdf}`. It shows the clip means of ours against the
labels, and the error by zoom. It contains no video frames, so it can be used in the paper.
