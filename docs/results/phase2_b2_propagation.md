# Phase 2 · B2: the pressing correction with the visible-player count from real video

**Notebook:** `colab/09_count_error_propagation.ipynb` · **Code:** `pitchlens.analytics.propagation`,
`pitchlens count-error-propagation` · **Run 1:** 2026-11-25, laptop (CPU, under 1 min)

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

## Run 1 (500 draws per broadcast; B1 counts from Drive `outputs/b1_visible/counts_tracks.parquet`)

**Count errors per clip** (78 held-out and test clips): raw −0.84 ± 0.86 players; calibrated −0.07 ± 0.87
players. The calibration is a = 0.943, b = 1.52, the same as B1.

### Active pressers, error after correction (%, over 360 broadcasts × 500 draws)

| Scenario | Mean | Mean abs. | 95th pct abs. | Within ±5% | Within ±10% |
|---|---|---|---|---|---|
| No correction | −21.77 | 21.77 | 29.87 | 0.0% | 2.2% |
| Corrected, true count (paper v1) | +0.02 | 3.05 | 7.29 | 81.4% | 100% |
| Corrected, video count raw, whole match | +4.71 | 5.19 | 11.31 | 51.7% | 89.8% |
| Corrected, video count raw, one clip | +4.97 | 6.19 | 17.14 | 52.7% | 81.8% |
| **Corrected, video count calibrated, whole match** | **+0.37** | **3.09** | **7.27** | **79.4%** | **100%** |
| Corrected, video count calibrated, one clip | +0.59 | 4.67 | 12.63 | 64.4% | 90.5% |

### Sensitivity: a fixed count error

| Count error (players) | −2 | −1.5 | −1 | −0.5 | 0 | +0.5 | +1 | +1.5 | +2 |
|---|---|---|---|---|---|---|---|---|---|
| Mean error (%) | +11.9 | +8.7 | +5.6 | +2.8 | 0.0 | −2.6 | −5.0 | −7.4 | −9.6 |
| Mean abs. error (%) | 11.9 | 8.7 | 5.9 | 3.9 | 3.05 | 3.6 | 5.2 | 7.4 | 9.6 |

Each player of count error shifts the corrected value by about 5–6%. This is the 4.1 pp slope divided by
(1 + bias), with bias around −0.2.

**Figure:** `outputs/b2_propagation/b2_propagation.{png,pdf}` on the laptop. It is a violin plot of the four
headline scenarios and contains no video frames.

## Findings

1. **With the calibrated count from real video, the correction is as good as with the true count.** For a whole
   match, the mean absolute error is 3.09% against 3.05%, the 95th percentile 7.27% against 7.29%, and every
   broadcast is within ±10%. The video adds almost nothing to the error.
2. **Calibration is necessary.** The raw undercount (about 0.84 players) makes the correction overshoot: +4.7% mean
   error, mean absolute error 5.2%, 95th percentile 11.3%. That is still far better than no correction (−21.8%),
   but not as good. The calibration is fitted on clips separate from those it is evaluated on.
3. **Short excerpts are noisier.** If the visible-player mean comes from a single 30-second clip, the mean absolute
   error rises to 4.7% and the 95th percentile to 12.6%. The correction is meant for whole matches, where clip
   errors average out.
4. **The "whole match" model assumes the clip errors of a match are independent.** If one camera style shifts all
   of a match's clips the same way, the truth lies between the two models. Even the worst case (one clip) keeps
   90.5% of broadcasts within ±10%, against 2.2% without correction.

**For the paper (real-video section):** on SN-GSR, the pipeline undercounts visible players by 0.7–1.1 per frame
(B1). After a calibration fitted on separate clips, plugging the video count into the correction gives a 3.1% mean
absolute error and a 7.3% 95th percentile for whole matches, the same as with the true count.
