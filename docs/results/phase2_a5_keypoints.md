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
