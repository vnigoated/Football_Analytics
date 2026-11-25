# Phase 2 · B3: a main-camera filter for replays and close-ups

**Notebook:** `colab/10_main_camera_filter.ipynb` · **Code:** `pitchlens.vision.shots` · **Run 1:** pending

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

## Results

Pending run 1.
