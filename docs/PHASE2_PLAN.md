# Phase 2 plan: from broadcast video to trustworthy pressing data

**Period:** December 2026 – June 2027 · **Status:** planned (written 2026-11-22)

Phase 1 is complete. It measured, validated and corrected the visibility bias of pressing metrics, all on
*simulated* broadcast views; the preprint is ready for arXiv. Phase 2 builds the computer-vision half of PitchLens
(TV video → tracking data) and uses it to close the paper's main gap: **no real broadcast video has been
processed yet**.

## Goals

1. **Pipeline:** turn broadcast clips into per-frame player positions on the pitch (metres), with team and track IDs.
2. **Benchmark:** report standard metrics on SoccerNet Game State Reconstruction (GS-HOTA and its parts).
3. **Bridge to the paper:** measure how accurately the *number of visible players* (the correction's input) can be
   obtained from real video, and propagate that error into the bias correction.
4. **Product:** possession, passes, xT, a minimap video and a dashboard; release v1.0.

## The key constraint

No open dataset has broadcast video **and** full tracking of all 22 players for the same match. So on real video we
cannot measure the pressing bias directly. Instead:

- SoccerNet-GSR has ground-truth pitch positions of the **visible** players: we validate the pipeline and the
  visible-player count against it.
- The bias itself stays measured on full tracking (IDSSE, Metrica) with cameras fitted to real broadcasts
  (SkillCorner), as in Phase 1.
- The link between the two is the visible-player count: we show it can be measured on video and how much its error
  widens the corrected estimate.

## Work packages

### 2A. Video → players on a minimap (December – January, Colab A100)

| # | Task | Output | Target / metric |
|---|---|---|---|
| A1 | Download SN-GSR train + valid to Colab; notebook that draws labels on frames and prints statistics (clips, frames, players and **visible players per frame**) | `colab/01_gsr_data.ipynb` | Data understood; label format documented |
| A2 | Fine-tune **RF-DETR** (Apache-2.0) for player, goalkeeper, referee, ball | checkpoint on Drive, `pitchlens.vision.detector` | mAP@50 on valid; ball recall reported separately |
| A3 | **Tracking** with ByteTrack (`trackers`, Apache-2.0; OC-SORT and SORT compared), tuned on half of valid, reported on the other half | `pitchlens.vision.tracker` | HOTA / IDF1 on valid |
| A4 | **Team assignment**: SigLIP embeddings → UMAP → k-means per clip; goalkeepers by position | `pitchlens.vision.teams` | Team accuracy on valid |
| A5 | **Pitch keypoints → homography**: 33-keypoint heatmap model (ResNet-34), labels derived from SN-GSR player positions, RANSAC homography (already in `pitchlens.geometry`), temporal smoothing | `pitchlens.vision.keypoints` | Reprojection error (m); share of frames with a valid homography |
| A6 | **End-to-end** run on SN-GSR valid and test; homographies checked and filled over time; roles (goalkeeper / referee) and left / right sides | `pitchlens video` command, `pitchlens.vision.pipeline`, `colab/06_pipeline.ipynb`, results table | **GS-HOTA**, compared with published baselines (SoccerNet baseline 22.26 on test) and Choi (2026), who reports 35.2 on 11 test sequences. No jersey numbers yet: also reported without jersey and position only. **Run 1: test GS-HOTA 23.50 (without jersey 52.61, position only 61.57)** |
| A7 | **Jersey numbers**: read the number on player crops and vote per track (oracle numbers lift held-out GS-HOTA 25.4 → 69.4); re-score A6 with the official SoccerNet evaluator | `pitchlens.vision.jersey`, `colab/07_jersey.ipynb` | GS-HOTA on valid held-out and test. **Run 1: official test GS-HOTA 30.52** (24.00 without numbers); reader weak (overfits 41 train numbers) -> run 2: pretrained text recogniser (PARSeq) + legibility filter |

### 2B. Real video → the paper (February – March)

| # | Task | Output |
|---|---|---|
| B1 | Visible players per frame from the pipeline vs. ground truth: bias, error, how it depends on zoom and replays | table + figure; `pitchlens.analytics.visible_count`, `colab/08_visible_players.ipynb`. **Run 1: bias -0.7 to -1.1 players raw, -0.32 / +0.09 after calibration (held-out / test)** |
| B2 | Propagate that error into the correction (Phase 1 `bias-correction`): corrected-error interval for real use | updated correction result; `pitchlens count-error-propagation`, `colab/09_count_error_propagation.ipynb`. **Run 1: calibrated video count, whole match: MAE 3.09% (true count 3.05%), p95 7.27%** |
| B3 | Replay / close-up detection (frames where the homography fails or the view is not the main camera) and their effect | filter + share of frames removed; `pitchlens.vision.shots`, `colab/10_main_camera_filter.ipynb` (SN-GSR is main camera only: measures false removals) |
| B4 | Demo: pressing and corrected pressing on real broadcast clips | figure for the paper |
| B5 | **Paper v2**: add the real-video section; submit to CVsports 2027 (if the deadline fits, ~March) or MLSA 2027 (~June) | updated `paper/` |

### 2C. Product (March – June)

| # | Task | Output |
|---|---|---|
| C1 | Ball possession from tracks (nearest player + ball speed); passes and carries | `pitchlens.events` |
| C2 | xT on detected actions (`pitchlens.analytics.xt` already exists) | per-player xT |
| C3 | Minimap video and a simple dashboard | `pitchlens render`, dashboard |
| C4 | Near-live mode: process 10-second chunks with a short delay | `pitchlens live` |
| C5 | Export to kloppy format; docs; Docker; PyPI; **v1.0** | release |

## Timeline

| Weeks | Work |
|---|---|
| Dec wk 1–2 | A1, A2 |
| Dec wk 3 – Jan wk 1 | A3, A4 |
| Jan wk 2–4 | A5, A6 |
| Feb | B1–B4 |
| Mar | B5 (CVsports if open), C1 |
| Apr – May | C2–C4 |
| Jun | C5, MLSA 2027 submission |

## Compute and storage

- **Colab A100** for training (A2, A5) and end-to-end runs (A6). Data on `/content` (fast, wiped per session),
  checkpoints and results on Google Drive.
- SN-GSR train + valid ≈ 21 GB; the laptop has ≈ 140 GB free if local copies are needed.
- Laptop for code, tests and analysis on saved results.

## Licences

- SoccerNet SN-GSR: GPL-3.0 (dataset card). Research use and training are fine; **do not redistribute** frames or
  videos. Check the licence before publishing trained weights.
- RF-DETR, ByteTrack (`supervision`): open-source permissive licences. Ultralytics YOLO stays optional (AGPL-3.0).

## Risks and fallbacks

| Risk | Fallback |
|---|---|
| Ball detection is weak (small, blurred) | Report it separately; interpolate short gaps; possession from player proximity |
| Homography fails on close-ups / replays | Detect and skip those frames (B3); report the share skipped |
| Colab session limits interrupt training | Resumable training, checkpoint every epoch to Drive |
| GS-HOTA far below published baselines | Use SoccerNet's public baseline components for the weak stage and focus the contribution on B1–B2 |
| CVsports 2027 deadline earlier than expected | Target MLSA 2027 (June) as planned |

## Definition of done for Phase 2

- `pitchlens video clip.mp4` produces positions, teams and a minimap video.
- GS-HOTA reported on SN-GSR with a reproducible notebook.
- The paper has a real-video section with the measured visible-player error and the corrected-error interval.
- v1.0 released with docs, Docker image and results table in the README.
