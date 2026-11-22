# Phase 2 · A1: SoccerNet-GSR data check

**Date:** 2026-11-22 · **Notebook:** `colab/01_gsr_data.ipynb` · **Code:** `pitchlens.data.gsr`

## Data

SoccerNet Game State Reconstruction (SN-GSR-2025, Hugging Face, GPL-3.0; not redistributed).
Each sequence is a 30-s broadcast clip (750 frames, 25 fps, 1920×1080) of Swiss Super League matches, with
per-frame boxes, track IDs, role (player / goalkeeper / referee / ball / other), team (left / right), jersey
number and the pitch position (metres, origin at the centre) of each box's bottom middle. Pitch-line annotations
(26 line classes) are also present and will be used for A5 (keypoints → homography).

Note: the zips have no `train/` / `valid/` folder inside; the notebook extracts each into its own folder.

## Statistics

| | train | valid |
|---|---|---|
| Sequences | 57 | 58 |
| Frames (all labelled) | 42,750 | 43,500 |
| Objects | 733,001 | 748,666 |
| Tracks | 1,427 | 1,426 |
| **Visible players per frame, mean** | **14.72** | **14.84** |
| median / p10 / p90 | 15 / 9 / 20 | 16 / 9 / 20 |
| Frames with the ball labelled | 94.1% | 93.0% |
| Players with a pitch position | 99.3% | 98.4% |
| Positions off the pitch (> 60 m / 40 m from centre) | 0.59% | 0.47% |

Visible players include goalkeepers and partly visible players. Every player and goalkeeper has a team label;
objects without a team are exactly the referees, the ball and "other".

**Valid split in detail**

| Role | Objects | | Team | Objects |
|---|---|---|---|---|
| player | 619,500 | | right | 326,679 |
| referee | 61,847 | | left | 318,841 |
| ball | 40,900 | | none | 103,146 (= referees + ball + other) |
| goalkeeper | 26,020 | | | |
| other | 399 | | | |

- Missing pitch position: 1.52% of objects (all other columns complete).
- Off-pitch positions: train 4,274 frames, valid 3,500 frames (one object each in valid, always the ball).
  Most affected valid clips: SNGS-024 (238), SNGS-037 (195), SNGS-051 (173), SNGS-055 (171), SNGS-028 (147).
- Plausible positions span the pitch (±52.5 m × ±34 m) plus a few metres outside the lines.

**Label format (version 1.3)**

- `info`: game id, clip id, action class around which the clip is cut (e.g. "Kick-off"), game time, frame rate,
  sequence length.
- `images`: `image_id`, `file_name`, `height`, `width`, `is_labeled`, `has_labeled_person`, `has_labeled_pitch`,
  `has_labeled_camera`, ignore regions.
- `annotations`, `supercategory` `object`: `track_id`, `category_id`, `attributes` (role, jersey, team), `bbox_image`
  (x, y, w, h, centre), `bbox_pitch` and `bbox_pitch_raw` (bottom left / middle / right, metres). We use
  `bbox_pitch` bottom middle.
- `annotations`, `supercategory` `pitch`: line annotations (26 classes such as "Middle line", "Circle central").
- `categories`: player 1, goalkeeper 2, referee 3, ball 4, pitch 5, camera 6, other 7.

**Saved outputs** (Google Drive `MyDrive/pitchlens/outputs/gsr/`, private): `summary.csv`,
`{train,valid}_{images,objects,visible_per_frame}.parquet`, `visible_players.png` (histogram and clip means
against the 14.65 line), `sample_frames.png` (labelled frames; contains SN-GSR images, do not publish).

## Findings

1. **Real broadcasts show 14.7–14.8 players on average**, against 14.65 predicted by the Phase 1 camera
   fitted to SkillCorner (A-League). A different league, a different label source and the same number: the
   simulated cameras used for the bias estimates are realistic in the quantity that drives the bias.
2. **Large variation between clips** (clip means from about 9 to 20 players). In Phase 1 terms this spans
   roughly the −10% to −29% range of active-presser bias, so the correction must be applied per clip or per
   broadcast, not as one global factor.
3. **All off-pitch positions are ball labels** (valid: 3,500 objects in 3,500 frames, about 8.6% of ball
   labels; worst clips SNGS-024, -037, -051). The ball's position is projected from its box bottom onto the
   ground plane, which is wrong when the ball is in the air. Player positions are clean. Consequences: ignore
   ball pitch positions outside the pitch, and treat ball position as reliable only near the ground (matters
   for possession in C1).

## Caveat

SN-GSR clips are chosen around annotated actions (e.g. kick-offs) and are not a random sample of match time,
so the visible-player distribution may differ slightly from full broadcasts. B1 compares the pipeline's count
with these labels, not with full matches.

## Next

A2: fine-tune RF-DETR on train (player, goalkeeper, referee, ball), evaluate mAP@50 on valid, with ball recall
reported separately.
