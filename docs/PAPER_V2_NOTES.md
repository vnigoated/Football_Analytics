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
| 4 | An off-the-shelf detector finds most people on broadcast frames | person recall 0.949, precision 0.903 (RF-DETR Medium, 896 px, run 1) | `docs/results/phase2_a2_detector.md`; Drive `outputs/detector/rfdetr_medium_896/` |
| 5 | Role labels cannot come from a single-frame detector | referee precision 0.28: kit colours that mark referees differ per match | same |
| 6 | Ball detection is the weak point | ball recall 0.574, precision 0.702 (centre within 10 px) | same |
| 7 | The pipeline can run faster than real time | 41 images/s at 896 px (batch 16) | same |

Run 2 (person + ball, 1120 px) is in progress; it replaces rows 4, 6 and 7 when done.

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
- **To record in each run:** GPU type, `torch` and `rfdetr` versions (printed by the notebook's install cell).
- Licences: SN-GSR images may not be published, so the paper's example figures must use our own drawings or
  permitted material, not SN-GSR frames.
