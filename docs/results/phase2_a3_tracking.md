# Phase 2 · A3: tracking people across frames

**Notebook:** `colab/03_tracking.ipynb` · **Code:** `pitchlens.vision.tracker` · **Run:** 2026-11-25, Colab A100
(12 CPUs)

Detector: A2 run 2 (RF-DETR Medium, person + ball, 1120 px), run once on every frame of SN-GSR valid (58 clips ×
750 frames) at score threshold 0.05; boxes cached on Drive (`outputs/tracking/detections_valid/`). Trackers and
metrics: Roboflow `trackers` 2.x (Apache-2.0); HOTA, CLEAR (MOTA) and Identity (IDF1) ported from TrackEval.
Ground truth: players, goalkeepers and referees (staff and ball excluded), boxes clipped to the image. Settings
were tuned on the 29 even-numbered clips (sorted by name) and are reported on the 29 others (**held out**).

## Results (held-out half, 29 clips)

| Tracker | HOTA | DetA | AssA | IDF1 | MOTA | ID switches |
|---|---|---|---|---|---|---|
| ByteTrack, default settings | 0.641 | 0.749 | 0.550 | 0.757 | 0.930 | 1,065 |
| ByteTrack, tuned | 0.642 | 0.748 | 0.553 | 0.759 | 0.929 | 1,060 |
| **OC-SORT + camera-motion compensation** | **0.647** | 0.740 | **0.567** | **0.767** | 0.917 | **871** |
| ByteTrack tuned, on ground-truth boxes (perfect detection) | 0.886 | 0.993 | 0.790 | 0.844 | 0.992 | 947 |

All 58 valid clips, ByteTrack: default HOTA 0.639, tuned 0.638 (the tuned settings do not generalise better than the
defaults).

Tuning (tuning half, 32 ByteTrack settings, 248 s on 12 CPUs): HOTA ranged only from 0.616 to 0.633. The one setting
that mattered was the high-confidence threshold (0.7 better than 0.5); the lost-track buffer (30 vs 90 frames) made
no difference. Best: activation 0.4, high-confidence 0.7, buffer 30, minimum IoU 0.1, minimum 1 consecutive frame.
Filling gaps by linear interpolation did not help (HOTA 0.6331 with no filling, 0.6327 with gaps up to 50 frames),
because the detector rarely misses a person for a few frames only.

OC-SORT with camera-motion compensation reads the frames (sparse optical flow), so it took 636 s for the 29 held-out
clips (about 34 frames/s on the CPU), against seconds for trackers on cached boxes alone.

## Findings

1. **Detection is good, identity is the weak part.** DetA 0.75 vs AssA 0.55: people are found and boxed well, but
   the same person often gets a new ID. The predicted number of tracks per clip is up to twice the true number
   (hardest clip SNGS-090: 58 tracks for 24 people).
2. **Most ID loss is not caused by the detector.** Even with perfect boxes, AssA is only 0.79 and there are 947 ID
   switches. On broadcast video, players leave the picture when the camera pans and come back later; SN-GSR keeps
   their identity, a motion-only tracker cannot. Example (SNGS-022): IDs run up to 49 by frame 750 for about 22
   people on the pitch.
3. **Camera-motion compensation is the one change that helped** (+0.006 HOTA, +0.017 AssA, −18% ID switches), as
   expected for a panning broadcast camera. Tuning ByteTrack's thresholds did not.
4. Known small issue: the trackers output nothing on a clip's first frame (a track is confirmed on its second frame);
   this costs at most 1/750 of the boxes.

**Decision:** use **OC-SORT with camera-motion compensation** in the pipeline. Re-linking a player who leaves and
re-enters the picture needs appearance and jersey numbers, not motion: this belongs to A4 (team assignment by
appearance) and the jersey / role step of A6, where GS-HOTA matches people by role, team and jersey rather than by
track ID. For B1 (visible players per frame) track IDs are not needed at all: detection quality (DetA, person
recall 0.958) is what counts.

## Still to record

Section 5 table (ByteTrack / OC-SORT / SORT with default settings on the tuning half) and the detection speed line.
