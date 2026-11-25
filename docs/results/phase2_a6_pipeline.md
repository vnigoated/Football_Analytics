# Phase 2 · A6: end to end, broadcast frames → game state (GS-HOTA)

**Notebook:** `colab/06_pipeline.ipynb` · **Code:** `pitchlens.vision.pipeline`, `pitchlens video` · **Run 1:** pending
(Colab A100)

## Pipeline

| Step | From | What |
|---|---|---|
| People boxes | A2 run 2 | RF-DETR Medium, 1120 px, score ≥ 0.05 (cached per clip) |
| Tracks | A3 | OC-SORT with camera-motion compensation |
| Homography per frame | A5 | 33 keypoints, RANSAC, at least 6 inliers |
| **Homography over time** | new | drop fits that disagree with their neighbours, fill short gaps, optional smoothing |
| Pitch positions | | box bottom middle through the frame's homography |
| Teams | A4 | shirt colour without lightness, one feature per track, k-means |
| **Roles and sides** | new | tracks fitting neither team: goalkeeper in a penalty area, referee elsewhere; the team further left on average is "left" |
| Jersey numbers | not yet | predicted as null |

**Homography over time.** Each homography is compared through where it sends nine fixed image points in the lower
part of the frame. A fit more than `outlier_m` metres (median) from the median of its neighbours within 6 frames is
dropped. Gaps up to `max_gap` frames are filled by interpolating those mapped points linearly and refitting. At the
start and end of a clip the nearest homography is held. An optional moving average over ±`smooth` frames reduces
jitter.

## Metric

GS-HOTA (Somers et al., CVPRW 2024) is HOTA with similarity `LocSim × IdSim`:

- `LocSim = exp(ln(0.05) · d² / τ²)`, where d is the pitch distance and τ = 5 m. Pairs scoring at or below 0.05 are
  never matched.
- `IdSim = 1` only if role, team and jersey number all match, otherwise 0. An attribute missing from the labels is not
  checked, for example a referee's team or an unreadable number. The ball is not scored.

Ground truth comes from SN-GSR `bbox_pitch`, in every labelled frame. Because we predict no jersey numbers, we also
report three relaxed variants: without jersey, role only, and position only. These mirror the SN-GSR paper's
ablation (Table 1). We also report oracle rows that replace one stage with the labels.

Published reference points:
- SoccerNet baseline: 22.26 on test and 18.05 on valid. Its position-only score is 42.65 on test.
- 2024 challenge winner: 63.81 on the challenge set.
- Choi (2026): 35.2 on 11 test sequences.

## Protocol

Settings are tuned on the even-numbered valid clips and reported on the odd-numbered clips, the same halves as A3
and A4:
1. Homography settings, chosen by position-only HOTA.
2. Role settings, chosen by HOTA without jersey.

Test is then run once with those settings.

## Results

Pending run 1.
