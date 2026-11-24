# Phase 2 · A4: team assignment by appearance

**Notebook:** `colab/04_teams.ipynb` · **Code:** `pitchlens.vision.teams` · **Run 1:** 2026-11-25, Colab A100 (in the
A3 session)

Method: tracks from ByteTrack (default settings) on the A3 cached detections; up to 10 crops per track from every
5th frame (21,924 crops over 58 clips); per clip, crops are clustered with k-means (after PCA to 16 dimensions) and
the two largest clusters are the teams, smaller ones "other"; each track takes its crops' majority label. Evaluation:
tracks matched to SN-GSR boxes per frame (IoU ≥ 0.5, Hungarian); player accuracy uses the better of the two
cluster→team mappings per clip; "other" counts as wrong for a player. Same tuning / held-out halves as A3.

Features: **colour** = Lab histogram of the shirt area (rows 15–50% of the box, middle 60% of the width), grass
pixels removed, 8 × 8 (a, b) bins + 4 lightness bins, Hellinger-normalised; **SigLIP** = `google/siglip-base-patch16-224`
image embedding of the top 60% of the box.

Time: tracks + crops 148 s, colour features 89 s, SigLIP features 111 s (A100), clustering seconds.

## Run 1: tuning half (29 clips)

| Feature | k | Player accuracy | Players as "other" | Worst clip | Clips < 90% | Referees as "other" | Keepers as "other" |
|---|---|---|---|---|---|---|---|
| **colour** | **2** | **0.889** | 0 | 0.537 | 9 | 0 | 0 |
| colour | 4 | 0.865 | 0.087 | 0.537 | 14 | 0.933 | 0.871 |
| colour | 3 | 0.853 | 0.098 | 0.471 | 13 | 0.785 | 0.735 |
| SigLIP | 3 | 0.653 | 0.256 | 0.389 | 25 | 0.334 | 0.281 |
| SigLIP | 2 | 0.644 | 0 | 0.504 | 27 | 0 | 0 |
| SigLIP | 4 | 0.533 | 0.390 | 0.152 | 28 | 0.515 | 0.592 |

**Held out (colour, k = 2): player accuracy 0.829**, 12 of 29 clips below 90%; all 58 clips 0.859.

## Findings

1. **Colour beats SigLIP by a wide margin** (0.89 vs 0.65). A general image embedding encodes pose, size and
   background as much as kit colour; clustering it per clip does not separate teams.
2. **Failures are whole clips, not scattered errors.** The six worst held-out clips (SNGS-040, 044, 046, 048, 054,
   058) all score 0.51–0.55, i.e. chance: there the two clusters split something other than the teams (likely
   lighting or crop size), while most other clips are near-perfect. Their consecutive numbers suggest one or two
   matches.
3. With k = 4, the small clusters catch 93% of referee and 87% of goalkeeper boxes, at the cost of 8.7% of player
   boxes; with k = 2 referees and goalkeepers are forced into a team.

## Diagnosis and fix (same session)

Crops of the failing clip SNGS-048 (one team in yellow, the other in green and white), grouped by the k = 2
clusters: **one cluster held the shaded crops and the other the sunlit crops**, each containing both teams. Half
the pitch was in shadow, and the lightness bins and the brightness of the colours dominated the distance.

Two changes, tested on the tuning half and confirmed on the held-out half:

* **colour without lightness**: drop the 4 lightness bins, keep the (a, b) colour histogram;
* **cluster per track**: average each track's crops (only crops ≥ 40 px tall when it has any) into one feature,
  weight each track by its number of crops, and cluster tracks instead of crops.

| Level | Lightness | k | Player accuracy (tuning) | Worst clip | Clips < 90% | Referees as "other" | Keepers as "other" |
|---|---|---|---|---|---|---|---|
| **track** | **no** | **2** | **0.956** | 0.854 | 2 | 0 | 0 |
| crop | no | 2 | 0.950 | 0.823 | 4 | 0 | 0 |
| track | no | 3 | 0.942 | 0.811 | 5 | 0.626 | 0.745 |
| crop | no | 3 | 0.928 | 0.770 | 7 | 0.694 | 0.752 |
| track | yes | 2 | 0.897 | 0.502 | 6 | 0 | 0 |
| crop | yes | 2 (run 1) | 0.889 | 0.537 | 9 | 0 | 0 |
| crop | no | 4 | 0.874 | 0.689 | 14 | 0.838 | 0.843 |

**Held out (track, no lightness, k = 2): player accuracy 0.945** (run 1: 0.829), worst clip 0.624, 3 of 29 clips
below 90% (run 1: 12). With k = 4 on the held-out half: 0.872 player accuracy, 82% of referees and 82% of
goalkeepers set apart, 8.4% of players wrongly set apart.

Dropping lightness is the main gain (+5 to +6 points); clustering per track adds about 1 point and lifts the worst
clips. These are now the defaults.

## Open

* **Referees and goalkeepers.** With k = 2 they are forced into a team. More clusters set most of them apart but
  also split off ~8% of players. Better: keep k = 2 for the teams and mark as "other" the tracks that are far from
  both team centres; goalkeepers can then be assigned to a team by position once pitch coordinates exist (A5).
* Three held-out clips are still below 90% (worst 0.62); to inspect.
* Teams are labelled A / B per clip; matching them to SN-GSR's left / right (which side a team defends) needs pitch
  positions (A5).
