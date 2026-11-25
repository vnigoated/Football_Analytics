# Journal step 4: the visible-player count on an independent dataset (SportsMOT football)

**Code:** `pitchlens.data.sportsmot`, `colab/11_sportsmot_count.ipynb` (detector A2 run 2, keypoints A5, tracker and
roles A6, unchanged) · **Run:** 2026-11-26, Colab (detection and tracking) and laptop (analysis)

## Why SportsMOT and not SN-Tracking-2023

All 106 labelled SN-Tracking-2023 clips (57 train, 49 test) are SoccerNet-GSR clips: each matched an SN-GSR clip on
clip start, action position and game time. So they would only repeat B1.

SportsMOT (Cui et al., ICCV 2023; Hugging Face `MCG-NJU/SportsMOT`, CC BY-NC 4.0) has 30 labelled football clips from
other broadcasts: 15 train and 15 val. Its frames are 1280×720, and its clips are 18–53 s long (20,171 frames in all).
Only players on the pitch are labelled: both teams and goalkeepers, without referees. That is the count the correction
uses. Nothing was trained or tuned on SportsMOT; B1's calibration (`labels = 0.943 · ours + 1.52`, fitted on the SN-GSR
tuning half) is applied unchanged.

## Results

| Set | Clips | Labels | Ours | Bias [95% CI] (players) | Clip MAE | Clip p95 | Clip r |
|---|---|---|---|---|---|---|---|
| SportsMOT, raw | 30 | 13.01 | 12.74 | **−0.26 [−0.54, −0.01]** | 0.70 | 1.62 | 0.90 |
| SportsMOT, B1 calibration | 30 | 13.01 | 13.53 | +0.53 [+0.26, +0.79] | 0.78 | 1.67 | 0.90 |
| SportsMOT val, pooled calibration (SN-GSR tune + SportsMOT train) | 15 | 12.93 | 13.35 | +0.42 [+0.06, +0.80] | 0.78 | 1.47 | 0.92 |
| SN-GSR held out + test, B1 calibration (reference) | 78 | 13.83 | 13.76 | −0.07 [−0.27, +0.13] | 0.62 | 1.78 | 0.95 |

**By zoom** (raw; median labelled player height): close +0.12, mid-close 0.00, mid-wide −0.27, wide −0.88 players.

**By number of labelled players** (raw): ≤ 10: +0.09; 11–14: −0.26; 15–18: −0.44; ≥ 19: −1.10.

This is the same pattern as on SN-GSR: the players we miss are small, or hidden in crowded scenes.

On 32% of the frames the A5 keypoints give no homography (720p, other stadiums). The count doesn't need one, and the
error on those frames is −0.04 players.

### What the count error does to the corrected pressing value

This is B2's propagation, using the 10-match table (400 team-broadcasts) and the SportsMOT clip errors:

| Count | Whole match: MAE / p95 | One clip: MAE / p95 |
|---|---|---|
| True count (paper v1) | 3.00% / 6.95% | — |
| SN-GSR video count, calibrated (B2, 10 matches) | 3.02% / 7.10% | 4.64% / 12.56% |
| **SportsMOT video count, raw** | **3.28% / 8.0%** | 4.83% / 12.15% |
| SportsMOT video count, B1 calibration | 3.58% / 8.7% | 4.89% / 11.38% |

## Conclusion

- **The count carries over to new broadcasts.** On 30 unseen clips the raw count is off by only −0.26 players, and
  the clip means correlate at r = 0.90.
- **B1's calibration does not carry over.** It adds about 0.8 players, which fits SN-GSR (−1.05 raw) but overshoots
  here (+0.53). Adding SportsMOT's own train clips to the fit barely moves it (+0.42). The offset therefore depends on
  the dataset: on how small or hidden players are labelled, and on how crowded the broadcasts are.
- **Either way the count error stays within about ±0.5 players** (≈ 2 pp of bias). On a whole match the corrected
  pressing error rises only from 3.0% to 3.3–3.6%.
- **For the paper:** report a count uncertainty of ±0.5 players across datasets, with an implied correction error of
  3.3–3.6% (p95 8–9%). Don't present the calibration as universal.
