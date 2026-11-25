# Journal step 3: camera model fitted to a second league (SoccerNet-GSR, Swiss Super League)

**Code:** `pitchlens.data.gsr_camera` (camera footprints from the labelled pitch positions),
`pitchlens.simulation.camera_fit` (`fps` parameter), `pitchlens.analytics.robustness.transfer_correction` ·
**Run:** 2026-11-25, laptop

```bash
pitchlens fit-camera-gsr --out configs/cameras/gsr          # 6 SN-GSR games (valid 2/3/5, test 7/8/11)
pitchlens visibility-bias --match <each of 10> --camera-json configs/cameras/gsr --tag gsr --sample-every 5 \
    --gt-dir outputs/pressing_baseline --out outputs/stadiums_gsr/idsse        # metrica: outputs/stadiums_gsr/metrica
pitchlens cross-league                                       # ~43 min for the 60 runs in total
```

## Method

SN-GSR provides no camera parameters. However, each person's labelled pitch position is an exact projection of the
bottom of their box. So for every frame we fit a homography to those positions and map the four image corners onto
the pitch. This gives the camera footprint, the same input SkillCorner provides for the A-League. The footprints are
rotated into SkillCorner's convention: camera at negative y, image-left at smaller x. A WedgeCamera model is then
fitted per game at 25 fps. We keep only frames whose four corners all project below the horizon and within 200 m.

The six Swiss cameras are then used to film all ten ground-truth matches (7 IDSSE and 3 Metrica), giving 120
team-broadcasts. Each is corrected with the line fitted on the 20 A-League cameras (400 team-broadcasts), leaving out
the broadcast's match.

## Fitted Swiss cameras

| Game | Camera distance (cam_y) | Median FOV | Lag | Real visible players (labels) | Simulated visible |
|---|---|---|---|---|---|
| test_11 | −65.8 m | 28.2° | 0.2 s | 15.40 | 12.84 |
| valid_3 | −63.3 m | 33.8° | 0.7 s | 16.01 | 12.66 |
| valid_2 | −52.1 m | 33.4° | — | 13.46 | 10.74 |
| test_7 | −52.4 m | 30.5° | 1.0 s | 13.17 | 10.62 |
| valid_5 | −55.5 m | 24.0° | — | 14.85 | 10.56 |
| test_8 | −51.0 m | 34.1° | — | 11.53 | 9.76 |

Pan R² is 0.90–0.96 where the lag was identified.

The simulated cameras show about 2.5–4 fewer players than the real broadcasts, though the order between games is
largely kept. The Swiss fit can follow only the ball, because SN-GSR doesn't say where the off-screen players are; the
A-League fit could also use player positions. The simulated Swiss cameras are therefore narrower than the real ones.
For the correction this mainly widens the test range: it pushes five of the six cameras below the A-League range of
11.7–16.9 visible players.

## Results

| | Value |
|---|---|
| Raw bias (Swiss cameras, 120 rows) | −37.7% (−27.5% to −44.9% per camera) |
| Slope, A-League (train) | 4.12 pp per visible player |
| Slope, Swiss cameras (test) | 5.10 pp per visible player |
| Corrected, A-League line transferred: MAE / p95 | 7.43% / 16.07% (mean −6.97%) |
| … rows inside the A-League visibility range (≥ 11.7, 40 rows) | **4.71%** (mean −3.94%) |
| … rows below it (80 rows) | 8.79% (mean −8.49%) |
| A-League quadratic instead of a line | 8.78% (p95 18.3%): curvature doesn't fix the extrapolation |
| Line pooled over the A-League and the other Swiss cameras (leave camera and match out) | **4.64%** (p95 11.54%) |

Per camera (corrected MAE): test_11 2.9%, valid_5 6.1%, valid_3 6.5%, valid_2 6.5%, test_8 9.7%, test_7 12.9%.

## Conclusion

- **The bias mechanism carries over.** With a camera from another league, the pressing count is still undercounted,
  and fewer visible players still means more bias. Below about 12 visible players the line gets steeper: 5.1 against
  4.1 pp per player.
- **The correction carries over within the visibility range it was fitted on:** 4.7% MAE, against 3.0% for the
  A-League cameras themselves.
- **It does not extrapolate:** below 11.7 visible players it under-corrects, by 8.5% on average.
- **Fitting on both leagues brings the error back to 4.6%.**

The correction should therefore be used only inside the visibility range it was fitted on, and the range should be
reported with it. The real Swiss broadcasts show 11.5–16.0 players, almost all inside the A-League range, so the
correction would apply to them.

**Caveats:**

- Only 6 Swiss games are fitted (valid/test; the train split is not downloaded).
- The ball-only fit makes the simulated cameras narrower than the real ones.
- Some frames were skipped because a corner projected beyond the horizon or 200 m. This affects test_8 most: 679
  frames remain.
