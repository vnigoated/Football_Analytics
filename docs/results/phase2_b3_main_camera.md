# Phase 2 · B3: a main-camera filter for replays and close-ups

**Notebook:** `colab/10_main_camera_filter.ipynb` · **Code:** `pitchlens.vision.shots` · **Run 1:** 2026-11-25, laptop (frames read from the SN-GSR zips)

## Question

The pressing correction assumes the wide main camera. Real broadcasts also show close-ups, replays and other
cameras. Frames from those views must be removed before the visible-player mean is computed. Can a simple filter
do that without throwing away normal frames?

## Filter

The filter needs no training data. It uses three cues:

1. **Grass share:** the share of pitch-green pixels (HSV hue 30–90, saturation ≥ 50, value ≥ 40). A frame needs at
   least `grass_min`, 0.3 by default.
2. **Person size:** the median height of the tracked people. A frame passes if it is at most `box_max`, 250 px by
   default. Frames without people pass.
3. **Shot cuts:** the Bhattacharyya distance between hue-saturation histograms of consecutive frames. A distance of
   at least `cut_min` (0.5 by default) starts a new shot. Each shot is kept or dropped by majority, so single noisy
   frames do not split a view.

## Limit of the evaluation

SN-GSR contains main-camera footage only: every labelled frame is marked as calibrated. On SN-GSR we can therefore
measure:
- how often the filter **wrongly drops** normal frames;
- whether dropping them changes the visible-player count.

We cannot measure how many real replays it catches. That would need broadcast video with shot labels (for example
SoccerNet-v2 camera annotations, whose videos are under NDA).

## Run 1 (2026-11-25, laptop; 80,250 frames of valid + test, read from the zips; 282 s)

The person size used here is the median height of the **labelled** player boxes, standing in for our tracked
boxes, because the tracks are on Drive. The A2 detector finds 96% of these people, so the two are close.

### The cues on main-camera footage

**Grass share:** median 0.64. The 5th percentile is 0.39 and the 1st percentile 0.21.

**Median player height:** median 103 px. The 95th percentile is 151 px, the 99th 189 px, and the maximum
**255 px**.

**Shot cuts:** none in any of the 107 clips. The largest colour change between consecutive frames is 0.16; the 99th
percentile is 0.048 and the 99.9th 0.075. A cut needs at least 0.4–0.6.

### Threshold sweep (36 settings)

The person-size threshold (200–400 px) and the cut threshold (0.4–0.6) never removed a frame. Only the grass
threshold matters:

| Grass minimum | Frames wrongly dropped | Clips affected | Players per frame (all → kept) | Count bias (all → kept) |
|---|---|---|---|---|
| **0.2** | **0%** | **0** | 14.08 → 14.08 | −0.809 → −0.809 |
| 0.3 | 1.87% | 2 | 14.08 → 14.03 | −0.809 → −0.812 |
| 0.4 | 3.74% | 4 | 14.08 → 14.00 | −0.809 → −0.823 |

**The frames dropped at 0.3** are two whole clips from the same stadium, SNGS-053 and SNGS-059, with grass shares
of 0.25 and 0.28 on average. Looking at them shows normal main-camera views in strong sunlight with hard shadows and
pale, yellowish turf. The grass colour range misses sunlit pale grass and dark shadow, so grass share falls to
0.17–0.28. These clips contain no cuts, so the vote drops each one as a whole.

**New defaults** (`main_view`): `grass_min = 0.2`, `box_max = 300` px, `cut_min = 0.5`. On SN-GSR these remove
**no** main-camera frames.

## Findings

1. **The filter does not harm main-camera footage.** With the new defaults no SN-GSR frame is removed. Even the
   stricter settings change the visible-player mean by at most 0.08 players and the count bias by at most
   0.014 players. That is less than 0.1 percentage points of pressing bias.
2. **The cut and size cues have wide margins.** Consecutive main-camera frames differ by at most 0.16, against 0.5
   needed for a cut. The tallest median player in a main-camera frame is 255 px, against a 300 px threshold; a
   close-up of a player is typically 500 px or more in a 1080p frame.
3. **Grass colour is the fragile cue.** Strong sun and shadows on pale turf push it down to about 0.2, so its
   threshold must stay low. A shot is therefore dropped as a close-up only when it has little grass *and* no cut
   back to the wide view; the size cue does most of the work.
4. **What is not measured:** how many real replays and close-ups the filter catches. That needs broadcast footage
   with shot labels. SN-GSR has none, by design. In the paper, the filter is reported as a conservative pre-step
   with a measured false-removal rate of 0% on 107 main-camera clips. Its recall is left to future work, for
   example SoccerNet-v2 camera-shot labels.

**Figure:** `outputs/b3_shots/b3_cues.{png,pdf}` on the laptop. It shows histograms of grass share and player size
on main-camera frames and contains no video frames. `dropped_examples.png` contains SN-GSR frames and stays private.
