# Phase 2 · A7: jersey numbers, and the official evaluator

**Notebook:** `colab/07_jersey.ipynb` · **Code:** `pitchlens.vision.jersey`, `pipeline.write_predictions` ·
**Run 1:** pending (Colab A100)

## Why

In A6, giving our tracks the labels' numbers lifted held-out GS-HOTA from 25.4 to 69.4, because 87% of player boxes
have a number. This step reads the numbers itself.

## Method

**Training data.** Torso crops of labelled players and goalkeepers that have a number, taken from SN-GSR train
(every 5th frame, boxes at least 50 px tall). The crop covers rows 5–65% of the box plus 10% of the width on each
side, resized to 96 × 128. The labels are per track, so many crops do not show the number.

**Network.** ResNet-34 (ImageNet weights) with one 20-way layer: the tens digit (0 = none, for one-digit numbers)
and the units digit. The two are combined into the probability of each number from 1 to 99. Training uses
colour jitter and a small shift or zoom, with no flips (a mirrored digit reads as a different one). It runs 12
epochs with AdamW and a one-cycle schedule. The best epoch is chosen by crop accuracy on the tuning half of valid.

**Vote per track.** Crops are read up to 40 times per track: every 2nd frame, boxes of at least 40 px, tallest
first. Crops whose best number has probability ≥ `min_conf` vote with their probabilities. A track gets the
winning number if at least `min_votes` crops voted, and no number otherwise. Both values are tuned on the tuning
half by GS-HOTA.

**Official rules.** We read the official evaluator (sn-trackeval `soccernet_gs.py`) and changed our GS-HOTA to
follow it:
- Team is scored only for players and goalkeepers.
- Jersey is scored only for players. Goalkeepers' numbers are not scored.
- A null number matches only a null number. A player whose number was never visible must be predicted without one.
- Only frames with labelled people, pitch and camera are scored.

A6 run 1 used a looser rule that skipped attributes missing from the labels, so section 3 of the notebook re-scores
A6 under the official rules. All predictions are also exported in the official JSON format and scored with
sn-trackeval itself.

## Results

Pending run 1.
