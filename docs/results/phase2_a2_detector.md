# Phase 2 · A2: player / ball detector (RF-DETR)

**Notebook:** `colab/02_detector.ipynb` · **Code:** `pitchlens.vision.coco`, `pitchlens.vision.metrics`

Training data: SN-GSR train, every 10th frame (4,275 images). Evaluation: SN-GSR valid, every 25th frame
(1,740 images, 1 per second). Our recall / precision use greedy one-to-one matching at IoU ≥ 0.5; the ball also
matches if the box centres are within 10 px (a ball box is only a few pixels wide). Score threshold 0.3.

## Run 1 (2026-11-23): 4 classes, 896 px

RF-DETR Medium (Apache-2.0, COCO-pretrained), resolution 896, effective batch 16, lr 1e-4, early stopping
(patience 4). Stopped after 10 epochs (~33 min); best EMA checkpoint at epoch 5.

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
know. Ignoring roles, 94.9% of people are found at 90.3% precision. The score plateaued from the first epoch
(0.353 → 0.364), so more epochs would not fix the role confusion.

**Decision:** detect only *person* and *ball*; assign roles (team A, team B, goalkeeper, referee) per clip in A4,
from appearance clustering within the clip plus position. This is also how the SoccerNet GSR baselines split the
problem.

## Run 2 (planned): person + ball, 1120 px

Same setup with `class_set="person_ball"` and resolution 1120 (to help the ball). Targets: person recall ≥ 0.95
at precision ≥ 0.92; ball recall above 0.57.
