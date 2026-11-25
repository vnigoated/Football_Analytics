# Phase 2 · B2: the pressing correction with the visible-player count from real video

**Notebook:** `colab/09_count_error_propagation.ipynb` · **Code:** `pitchlens.analytics.propagation`,
`pitchlens count-error-propagation` · **Run 1:** pending

## Question

Paper v1 corrects a broadcast's active-presser value using the true mean number of visible players. Over 360
broadcasts (20 stadium cameras × 9 matches × 2 pressing teams), each corrected with a line fitted without its own
camera and match:
- mean absolute error 3.05%;
- 95th percentile of the absolute error 7.29%;
- 4.14 percentage points (pp) of bias per visible player.

On real video the count comes from the pipeline (B1), with an error. How large is the corrected error then?

## Method

B1's per-clip count errors come from the held-out half of valid and from test: 78 real 30-second clips. They are
used raw, and after B1's calibration fitted on the tuning half. For every broadcast and every draw, a count error
is added to the true visible-player mean, and the same leave-camera-and-match-out correction is applied. Both
pressing teams of a broadcast share one draw, because it is the same video.

Two error models give the bounds:
- **Whole match:** the broadcast mean averages 180 clip errors, the length of a 90-minute match. Only the systematic
  part remains.
- **One clip:** the broadcast is off by a single clip's error. This is the worst case, with no averaging at all.

A sensitivity table gives the corrected error for a fixed count error from −2 to +2 players.

**Check:** with no count error, the code reproduces paper v1 exactly (0.02% mean, 3.05% MAE, 7.29% p95).

## Results

Pending run 1.
