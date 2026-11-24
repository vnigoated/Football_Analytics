# Paper v2 notes: what Phase 2 adds and where the evidence is

Paper v1 (arXiv preprint, `paper/`) uses simulated broadcast views only. Paper v2 adds a real-video section.
This file collects every Phase 2 result that may go into it, with its source, so nothing has to be reconstructed
from chat logs or Colab sessions. Update it after each step.

## Results so far

| # | Claim for the paper | Number | Evidence |
|---|---|---|---|
| 1 | Real broadcasts show about as many players as the fitted camera predicts | 14.72 (train) / 14.84 (valid) visible players per frame vs 14.65 predicted | `docs/results/phase2_a1_gsr_data.md`; Drive `outputs/gsr/summary.csv`, `visible_players.png` |
| 2 | Visibility varies strongly between broadcasts, so correction must be per clip | clip means about 9 to 20 players; frame p10 / p90 = 9 / 20 | same |
| 3 | Labelled ball pitch positions are unreliable in the air | all off-pitch positions are ball labels; 8.6% of valid ball labels | same |
| 4 | A fine-tuned detector finds almost all people on broadcast frames | person recall 0.958, precision 0.955 (RF-DETR Medium, person + ball, 1120 px, run 2) | `docs/results/phase2_a2_detector.md`; Drive `outputs/detector/rfdetr_medium_1120_person_ball/` |
| 5 | Role labels cannot come from a single-frame detector | run 1 (4 classes): referee precision 0.28, kit colours that mark referees differ per match; dropping roles raised person precision 0.903 → 0.955 | same; run 1 in `outputs/detector/rfdetr_medium_896/` |
| 6 | Ball detection is the weak point | ball recall 0.605, precision 0.719 (centre within 10 px; run 1 at 896 px: 0.574) | same |
| 7 | Detection speed | 14.7 images/s on an L4 at 1120 px (batch 16, not optimised); offline use, not yet real time | same |
| 8 | People can be tracked; identity across camera pans is the hard part | held-out HOTA 0.647 (OC-SORT + camera-motion compensation), DetA 0.74, AssA 0.57; with perfect boxes AssA only 0.79 | `docs/results/phase2_a3_tracking.md`; Drive `outputs/tracking/tracking_results.json` |
| 9 | Teams can be found per clip from shirt colour, without training | held-out player team accuracy 0.945 (colour without lightness, per track, k = 2); a general image embedding (SigLIP) only 0.65; lightness split clips into sun / shade | `docs/results/phase2_a4_teams.md`; Drive `outputs/tracking/teams_*` |
| 10 | Player pitch positions from broadcast frames, trained with labels derived for free from SN-GSR | median error 0.61 m, mean 0.94 m, 90% within 2 m, in 84.7% of frames (6+ keypoint inliers; 4-inlier fits were 86% wrong) | `docs/results/phase2_a5_keypoints.md`; Drive `outputs/keypoints/` |

## Still needed for the real-video section

- **B1:** visible-player count from the pipeline vs SN-GSR labels (bias and error per clip).
- **B2:** that error propagated into the correction: the corrected-error interval for real use.
- **B3:** share of frames removed as replays or close-ups.
- **A6:** end-to-end GS-HOTA on SN-GSR (comparison: Choi 2026 reports 35.2 on 11 test sequences).
- One figure: real visible-player distribution (SN-GSR) against the simulated cameras.

## Reproducibility details to report

- Data: SN-GSR-2025 from Hugging Face (GPL-3.0, not redistributed); frames sampled every 10th (train) and every
  25th (valid).
- Detector: RF-DETR Medium (Apache-2.0), COCO-pretrained; settings in `phase2_a2_detector.md`.
- Matching rule for our metrics: IoU ≥ 0.5, ball also within 10 px centre distance; score threshold 0.3.
- Run 2: NVIDIA L4 24 GB, Python 3.13, torch 2.14.1+cu130, rfdetr 1.12.1, 10 epochs in ~93 min. **To record in each run:** GPU type, `torch` and `rfdetr` versions (printed by the notebook's install cell).
- Licences: SN-GSR images may not be published, so the paper's example figures must use our own drawings or
  permitted material, not SN-GSR frames.
