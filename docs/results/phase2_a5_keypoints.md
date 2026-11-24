# Phase 2 · A5: pitch keypoints → homography → player positions

**Notebook:** `colab/05_keypoints.ipynb` · **Code:** `pitchlens.vision.keypoints` · **Run 1:** 2026-11-25, Colab A100
(same session as A3 / A4)

## Labels without annotation work

SN-GSR gives each labelled person a pitch position for the bottom middle of their box, computed with the
annotators' camera model. Fitting a homography (RANSAC, 0.5 m) to 6+ people per frame gives a ground-truth
image→pitch homography; projecting 33 pitch landmarks (29 line intersections, spots and circle points, plus the 4
points where the penalty arcs meet the box line) through its inverse gives their pixel positions.

The two position fields differ (first 10 valid clips, every 25th frame):

| Field | Frames with a homography | Median residual |
|---|---|---|
| `bbox_pitch_raw` | 276 / 300 | **0.002 m** |
| `bbox_pitch` | 291 / 300 | 0.107 m |

The raw field is an exact plane projection (residual ~0), so it is used. Labels built: **train 4,155 frames**
(every 10th; 120 without enough people), **valid 1,625** (every 25th); 8.8–9.1 visible landmarks per frame, 98.4–98.8%
of frames with 4 or more.

## Network and training

ResNet-34 (ImageNet weights) + 3 transposed-convolution stages → 33 heatmaps at stride 4 (input 960 × 544, output
240 × 136, Gaussian σ = 2 heatmap cells), MSE weighted 21× near keypoints, AdamW, one-cycle lr 1e-3, batch 16,
colour jitter only (no flips: left and right landmarks differ), bfloat16. 30 epochs in about 30 min on the A100;
validation loss lowest at **epoch 8** (0.00030), then flat while training loss kept falling: the epoch-8 weights are
used. Decoding: peak per heatmap above 0.3, 3 × 3 weighted refinement; RANSAC homography (1 m) with 4+ keypoints.

## Results (SN-GSR valid, 1,625 frames, 26,626 people)

| | Frames with homography | Median error | Mean | 90th pct | Within 1 m | Within 2 m |
|---|---|---|---|---|---|---|
| **Detected keypoints** | **98.8%** | **0.69 m** | 4.71 m | 3.23 m | 64.6% | 83.8% |
| Ground-truth homography (ceiling) | 100% | 0.01 m | 0.03 m | 0.07 m | 99.8% | 99.9% |

Keypoint localisation: median 8.7 px (in the 1920 × 1080 frame). Speed 27 frames/s including image loading.

Median per clip is about 1 m or less except **SNGS-055 (17 m)**, followed by SNGS-024 (1.8 m) and SNGS-023 (1.3 m).

## Findings

1. **Typical positions are good** (median 0.69 m), but the **mean (4.7 m) is driven by a few failed frames**
   where the homography is wrong altogether.
2. Failure seen in the samples (SNGS-083 frame 1): the detected landmarks lie almost on one line (the halfway line),
   so 4 inliers fit a degenerate homography that puts every player on that line. Such fits must be rejected and the
   frame filled from its neighbours (temporal smoothing, A6).
3. Overfitting after epoch 8 suggests more augmentation (scale / crop) or more training frames (every 5th) could
   help further.

## Which frames fail (diagnosis, same session)

Frames with a homography, by number of RANSAC inliers ("bad" = median player error > 2 m):

| Inliers | Frames | Bad | Median error |
|---|---|---|---|
| 4 | 125 | **85.6%** | 13.4 m |
| 5 | 104 | 25.0% | 1.23 m |
| 6 | 209 | 1.9% | 0.55 m |
| 7 | 210 | 9.0% | 0.74 m |
| 8 | 281 | 8.5% | 0.65 m |
| 12–17 | 80 | 0–6.5% | 0.30–0.87 m |

By spread of the detected landmarks on the pitch (standard deviation along the minor axis): 3–5 m: 49% bad;
over 5 m: 9–11% bad.

**Cause:** four points always fit a homography exactly, so a 4-inlier fit is never checked. SNGS-055 (the 17 m clip)
shows the typical case: the camera stays in midfield, where the only landmarks are the halfway line (one straight
line) and a few centre-circle points; the network finds them correctly, but they do not determine the view.

**Fix:** require a minimum number of inliers; rejected frames get their homography from neighbouring frames (A6). Longer term: more landmarks on the centre circle and along the lines, so
midfield views have enough points.

### Minimum number of inliers (same predictions, valid)

| Min. inliers | Frames with homography | Median error | Mean | 90th pct | Within 1 m | Within 2 m |
|---|---|---|---|---|---|---|
| 4 (run 1) | 98.8% | 0.69 m | 4.71 m | 3.23 m | 64.6% | 83.8% |
| 5 | 91.1% | 0.64 m | 1.21 m | 2.17 m | 68.7% | 88.8% |
| **6 (default)** | **84.7%** | **0.61 m** | **0.94 m** | **1.96 m** | **70.6%** | **90.4%** |
| 7 | 71.8% | 0.63 m | 0.97 m | 2.02 m | 69.5% | 89.9% |

**Chosen: 6.** The mean error falls from 4.71 m to 0.94 m and 90% of people are placed within 2 m; 15% of frames
get no homography of their own. With 7, more frames are lost without any gain.

**A5 result for the paper:** player pitch positions from single broadcast frames with a median error of 0.61 m
(mean 0.94 m, 90% within 2 m) in 84.7% of frames, from a network trained only on labels derived from SN-GSR player
positions.
