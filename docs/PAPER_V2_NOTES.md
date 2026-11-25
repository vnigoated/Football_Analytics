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
| 11 | End to end, broadcast → game state, above the SoccerNet baseline without jersey numbers | test GS-HOTA 23.50 (baseline 22.26), without jersey 52.61, position only 61.57 (baseline 42.65); valid held-out 25.38 / 61.66 / 70.10; oracle jersey → 69.35 | `docs/results/phase2_a6_pipeline.md`; Drive `outputs/pipeline/` (own GS-HOTA implementation: re-check with the official evaluator) |
| 12 | Calibration over time: checking, filling and smoothing homographies | +7.3 position-HOTA points; frames with homography 80.4% → 92.2% | same |
| 13 | Visible-player count from real video (B1 preview) | 13.52 vs 14.85 labelled per frame (−1.33), within ±1 in 70.6% of frames | same; `visible_players_held_out.csv` |
| 14 | Official SoccerNet evaluator: full pipeline with jersey numbers | **test GS-HOTA 30.52** (DetA 15.89, AssA 58.65; baseline 22.26), valid 28.30; without numbers 24.00 test; our implementation agrees within 0.6 points | `docs/results/phase2_a7_jersey.md`; Drive `outputs/jersey/official_*.txt` |
| 15 | Jersey numbers are the bottleneck | with labelled numbers on our tracks 52.15 (test) / 59.79 (valid held-out); own reader reads 17-21% of numbered tracks (41 numbers in train -> overfits) | same |
| 16 | Jersey numbers with a pretrained text recogniser fine-tuned on 1,507 crops | **official test GS-HOTA 35.54** (valid 36.09; DetA 20.97, AssA 60.25); reads 32% of numbered tracks vs 20% (run 1), wrong numbers 19% vs 62% | `docs/results/phase2_a7_jersey.md` run 2; Drive `outputs/jersey/official_*_pitchlens_a7r2.txt` |
| 17 | Self-training the number reader (round 2: 6,296 crops the round-1 reader sees a number in) | **official test GS-HOTA 39.77** (valid 41.63; DetA 26.47, AssA 59.76); numbered tracks read 55% (round 1: 32%) | `docs/results/phase2_a7_jersey.md` run 3; Drive `outputs/jersey/official_*_pitchlens_a7r3.txt`, `parseq_ft2.pt` |
| 18 | B1: the pipeline counts visible players with a small, correctable undercount | raw -1.05 [-1.41,-0.75] (held-out) / -0.72 [-0.95,-0.51] (test) players per frame; after a calibration fitted on separate clips -0.32 [-0.68,-0.01] / +0.09 [-0.13,0.29] = -1.3 / +0.4 pp of pressing bias (correction error itself 3.1%); clip r 0.95-0.96; undercount grows with wide shots (-1.35) and crowds (-1.75 at 19+) | `docs/results/phase2_b1_visible_players.md`; Drive `outputs/b1_visible/` (figure without frames) |
| 19 | B2: with the calibrated video count the correction is as good as with the true count | whole match: mean abs. error 3.09% (true count 3.05%), p95 7.27% (7.29%), 100% within +-10%; raw video count 5.19% / 11.31% (overshoot +4.7%); single 30-s clip 4.67% / 12.63%; about 5-6% per player of count error | `docs/results/phase2_b2_propagation.md`; laptop `outputs/b2_propagation/` (figure without frames) |
| 20 | B3: a main-camera filter (grass share, player size, shot cuts) removes no main-camera frames | 0% of 80,250 SN-GSR frames dropped with grass >= 0.2, player <= 300 px, cut >= 0.5; main-camera margins: max player 255 px, max frame-to-frame change 0.16; stricter grass (0.3-0.4) wrongly drops 1.9-3.7% (sunny pale-turf stadium) with < 0.1 pp effect; recall on real replays not measurable on SN-GSR | `docs/results/phase2_b3_main_camera.md`; laptop `outputs/b3_shots/` |

## Still needed for the real-video section

- ~~B1~~ done (row 18).
- ~~B2~~ done (row 19).
- ~~B3~~ done (row 20; recall on real replays remains open).
- **A6 follow-up:** re-score with the official SoccerNet evaluator; jersey numbers (A7).
- One figure: real visible-player distribution (SN-GSR) against the simulated cameras.

## Reproducibility details to report

- Data: SN-GSR-2025 from Hugging Face (GPL-3.0, not redistributed); frames sampled every 10th (train) and every
  25th (valid).
- Detector: RF-DETR Medium (Apache-2.0), COCO-pretrained; settings in `phase2_a2_detector.md`.
- Matching rule for our metrics: IoU ≥ 0.5, ball also within 10 px centre distance; score threshold 0.3.
- Run 2: NVIDIA L4 24 GB, Python 3.13, torch 2.14.1+cu130, rfdetr 1.12.1, 10 epochs in ~93 min. **To record in each run:** GPU type, `torch` and `rfdetr` versions (printed by the notebook's install cell).
- Licences: SN-GSR images may not be published, so the paper's example figures must use our own drawings or
  permitted material, not SN-GSR frames.

## Draft status

- 2026-11-25: real-video section drafted in `paper/main.tex` (Section 5, Table 4, Fig. 4) from rows 1-20 (branch `paper/v2-real-video`).
