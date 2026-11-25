# Phase 2 · B1: counting visible players on real video

**Notebook:** `colab/08_visible_players.ipynb` · **Code:** `pitchlens.analytics.visible_count` · **Run 1:** pending

## Question

The Phase 1 correction needs one input per broadcast: the mean number of visible players per frame (both teams,
goalkeepers included, no referees). Its slope is about 4.1 percentage points of active-presser bias per player.
How accurately does our pipeline measure that number on real broadcast video?

## Method

**Data.** Every labelled frame of SN-GSR valid and test. The labels count the labelled players and goalkeepers in
each frame, which is A1's definition: 14.84 per frame on valid.

**Two pipeline counts:**
- **tracks:** every tracked person whose role is player or goalkeeper. No homography is needed.
- **game state:** only those that also received a pitch position, as in A6.

**Calibration.** A linear fit `labels ≈ a · ours + b` on the tuning half of valid only. It is reported both with and
without the fit, on the held-out half and on test.

**Error measures:**
- the mean error (bias), with a 95% interval from resampling whole clips;
- the frame-level mean absolute error;
- the clip-level error of each 30-second clip mean.

**Error sources.** Errors are broken down by zoom (median labelled player height in pixels), by how many players the
labels show, and by the homography state of the frame (own / filled / held / none).

**Translation to pressing.** An error of δ players in a broadcast's mean shifts the predicted bias by 4.1 · δ
percentage points.

## Results

Pending run 1.
