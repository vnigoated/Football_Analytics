# Phase 2 · A2: player / ball detector (RF-DETR)

**Notebook:** `colab/02_detector.ipynb` · **Code:** `pitchlens.vision.coco`, `pitchlens.vision.metrics`

Training data: SN-GSR train, every 10th frame (4,275 images). Evaluation: SN-GSR valid, every 25th frame
(1,740 images, 1 per second). Our recall / precision use greedy one-to-one matching at IoU ≥ 0.5; the ball also
matches if the box centres are within 10 px (a ball box is only a few pixels wide). Score threshold 0.3.

## Run 1 (2026-11-23): 4 classes, 896 px

RF-DETR Medium (Apache-2.0, COCO-pretrained), resolution 896, effective batch 16, lr 1e-4, early stopping
(patience 4). Stopped after 10 epochs (~33 min); best EMA checkpoint at epoch 5.

Training details (from the RF-DETR log): 34.1 M trainable parameters (LW-DETR architecture, windowed DINOv2-small
backbone, patch 16, 300 queries, 4 decoder layers); detection head re-initialised from the 90-class COCO checkpoint
to 4 classes; bfloat16 mixed precision; square resize to 896 with multi-scale training (scale 1056 also used);
EMA weights selected as best (`checkpoint_best_total.pth`); 2,680 optimiser steps. Final training loss 4.92
(classification 0.49, box L1 0.010, GIoU 0.21). GPU and library versions were printed by the install cell but not
recorded for run 1; record them for run 2.

**RF-DETR COCO metrics (valid):** mAP@50 0.626, mAP@50:95 0.364, mAP@75 0.381, mAR 0.527.

| Class | Ground truth | Predicted | Recall | Precision |
|---|---|---|---|---|
| player | 24,860 | 23,205 | 0.852 | 0.913 |
| goalkeeper | 1,037 | 957 | 0.753 | 0.816 |
| referee | 2,477 | 5,663 | 0.637 | **0.278** |
| ball | 1,636 | 1,337 | 0.574 | 0.702 |
| **person, any role** | 28,374 | 29,825 | **0.949** | **0.903** |

Speed: 41 images/s at 896 px (batch 16, including image loading), faster than real time (25 fps).

**Finding: the detector finds people well; the role label is the problem.** Referee precision is 0.28 because
players whose kit resembles a referee kit (e.g. yellow) are labelled "referee", while referees in an unusual kit
are missed. The kit colour that marks a referee changes from match to match, which a single-frame detector cannot
know. Example (valid clip SNGS-059, Young Boys in yellow and black): the home players were predicted as
"referee", and the actual referee, dressed in black, was not detected; in other clips the yellow-clad referees were
labelled correctly. Ignoring roles, 94.9% of people are found at 90.3% precision. The score plateaued from the first epoch
(0.353 → 0.364), so more epochs would not fix the role confusion.

**Decision:** detect only *person* and *ball*; assign roles (team A, team B, goalkeeper, referee) per clip in A4,
from appearance clustering within the clip plus position. This is also how the SoccerNet GSR baselines split the
problem.

## Run 2 (2026-11-24): person + ball, 1120 px

Same setup as run 1 with `class_set="person_ball"` (player, goalkeeper and referee merged into *person*) and
resolution 1120 (positional encoding 70 × 70 patches), to help the ball. NVIDIA L4 (24 GB), batch 2 × gradient
accumulation 8 (effective batch 16 as in run 1). Targets set before the run: person recall ≥ 0.95 at precision
≥ 0.92; ball recall above 0.57. **All met.**

| Class | Ground truth | Predicted | Recall | Precision | Run 1 (896 px, 4 classes merged) |
|---|---|---|---|---|---|
| person | 28,374 | 28,461 | **0.958** | **0.955** | 0.949 / 0.903 |
| ball | 1,636 | 1,376 | **0.605** | **0.719** | 0.574 / 0.702 |

Same valid images (1,740), same matching rule and score threshold 0.3 as run 1, so the two runs are directly
comparable. Dropping roles mainly raised person precision (+5.2 points): run 1's extra "referee" boxes on players
were duplicates of correct player boxes. The higher resolution gave the ball +3.1 points of recall.

Sample predictions (valid): the referee in black in SNGS-059, missed in run 1, is now found; coaches and staff on
the touchline are correctly not detected (they are not labelled in SN-GSR); a ball near the far touchline is found in
SNGS-096.

Speed: 14.7 images/s on the L4 at 1120 px (batch 16, including image loading, model not exported/optimised for
inference). This is below real time (25 fps) but enough for the offline pipeline, and run 1's 41 images/s was on a
different GPU at lower resolution, so the two speeds are not comparable. The paper reports speed with GPU and
resolution.

Still to record from this run: RF-DETR COCO mAP (from `metrics.csv` in the Drive run folder), number of epochs
and training time, and the torch / rfdetr versions.

Checkpoint: Drive `outputs/detector/rfdetr_medium_1120_person_ball/checkpoint_best_total.pth`. **This is the
detector used from A3 on.**

**Remaining weakness: the ball.** About 40% of labelled balls are missed per frame. A3 tracking and later
interpolation can fill short gaps; a dedicated ball model (higher resolution crops or tiling) is an option if the
ball turns out to matter for B1 (it does not: B1 counts players).
