# arXiv submission metadata (copy into the form at https://arxiv.org/submit)

**Upload:** `paper/arxiv_submission.zip` (build with `python paper/make_arxiv.py path/to/llncs.cls`).

## Title

How Much Can You Trust Broadcast Pressing Data? Visibility Bias in Football Tracking Metrics

## Authors

Varun Inamdar (Vishwakarma University)

## Abstract

Tracking data derived from broadcast video is cheap and widely available, but the main camera shows only about 14 of the 22 players at a time. We measure how this partial view biases pressing metrics. Starting from full optical tracking of 7 Bundesliga and 2. Bundesliga matches (and 2 Metrica matches as a replication), we hide every player a broadcast camera would not show and recompute pressing intensity. The camera is not assumed: we fit a physical camera model to the on-pitch footprints of 20 real broadcasts (SkillCorner) and validate it on 10 held-out matches, where it predicts which players are on screen with 87.1% agreement and reproduces the mean number of visible players exactly (14.65). Averaged over 20 stadium cameras, broadcast tracking undercounts a team's active pressers by 22.6% [95% CI 19.8, 25.2] (replication: 19.0% [16.1, 22.0]); for an individual broadcast the undercount ranges from 10% to 29% and is almost perfectly explained by the number of visible players (r = 0.98, about 4 percentage points per player). Pressure on the ball carrier is nearly unbiased (-1.7%), but grows with camera lag (r = -0.98): even 0.2 s of lag increases it about fourfold. Both drivers can be measured on the broadcast itself. A simple correction based on the number of visible players reduces the mean absolute error in active pressers from 21.8% to 3.1% on broadcasts whose stadium camera and match were both held out. Code and fitted cameras are open source.

## Comments

11 pages, 3 figures, 2 tables. Code: https://github.com/vnigoated/Football_Analytics

## Categories

- Archive: **Computer Science (cs)**
- Primary: **cs.CV** (Computer Vision and Pattern Recognition); the endorsement must be for cs.CV
- Cross-list: **stat.AP** (Statistics – Applications). Optional: cs.LG (Machine Learning)
- Cross-lists need no extra endorsement; moderators may move or drop one, which is normal.

## Licence

**arXiv.org perpetual, non-exclusive license** (the default). It keeps you free to publish the paper later in a
workshop or journal under the publisher's own licence.

## Checklist before you press Submit

- [ ] Repository is public (the paper links to it).
- [ ] arXiv account uses your real name; affiliation Vishwakarma University.
- [ ] **Endorsement for cs.CV**: the endorser must be able to endorse in cs.CV (arXiv shows this on the
      endorsement page). Send them the endorsement code arXiv gives you.
- [ ] Check the generated PDF preview on arXiv carefully (tables, figures, references) before confirming.
