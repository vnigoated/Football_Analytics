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

## Run 1 (2026-11-25, Colab A100, same software as A6)

### A6 re-scored with the official rules (our implementation)

| | GS-HOTA | Without jersey | Role only | Position only |
|---|---|---|---|---|
| valid, held-out half | 25.24 | 61.01 | 66.81 | 69.47 |
| valid, all | 24.40 | 60.45 | 63.69 | 66.51 |
| test | 23.54 | 53.02 | 58.94 | 62.25 |

These are within 0.6 points of the looser A6 run 1 rule.

### Number reader

**Training data.** 101,169 crops from 57 train clips. They contain only **41 different numbers**; the most common
are 10, 9, 14, 8 and 4. Building the crops took 55 s.

**Training** (12 epochs, about 1 min each):
- Training loss fell from 2.02 to 0.04.
- Crop accuracy on the tuning half stayed at **0.06–0.11**, best 0.114 at epoch 6.
- So the network **overfits**. With few players and numbers, it can learn who the player is (kit, face) rather than
  the digits.

**Reading.** Up to 40 crops per track on all 107 clips took 156 s.

**Reader on labelled tracks** (valid held-out half, 564 player tracks):

| min_conf | Track right (incl. "no number") | Numbered tracks read right | Wrong number given |
|---|---|---|---|
| 0.3 | 0.174 | 0.202 | 0.823 |
| 0.5 | 0.183 | 0.206 | 0.787 |
| 0.7 | 0.209 | 0.206 | 0.707 |
| 0.9 | 0.227 | 0.171 | 0.450 |

**Vote** (tuning half, GS-HOTA): best `min_conf = 0.9`, `min_votes = 2`, giving 27.61 against 25.0–26.1 for the
looser settings. Only very sure reads help: a wrong number costs as much as no number.

### GS-HOTA with numbers

**Our implementation:**

| | GS-HOTA | With the labels' numbers on our tracks |
|---|---|---|
| A7, valid held-out half | 28.16 | 59.79 |
| A7, all valid | 27.89 | 59.66 |
| **A7, test** | **29.92** | 52.15 |

**Official SoccerNet evaluator** (sn-trackeval, labels version 1.3; all clips of each split):

| | GS-HOTA | DetA | AssA | LocA |
|---|---|---|---|---|
| A6 (no numbers), valid | 24.75 | 9.52 | 64.34 | 92.50 |
| A7 (numbers), valid | 28.30 | 12.56 | 63.80 | 92.40 |
| A6 (no numbers), **test** | **24.00** | 9.72 | 59.27 | 92.07 |
| A7 (numbers), **test** | **30.52** | 15.89 | 58.65 | 92.69 |

**The two implementations agree within 0.6 points** (test A7: ours 29.92, official 30.52; A6: 23.54 vs 24.00).
Our faster evaluation can therefore be used for tuning, and the official numbers are reported.

## Findings

1. **Official test GS-HOTA 30.52.** That is 8 points above the SoccerNet baseline (22.26) and below Choi (2026,
   35.2 on 11 test clips) and the 2024 challenge winner (63.81).
2. **The reader is weak but already useful:** +6.5 points on test (24.00 → 30.52), helped by voting only on very
   sure reads.
3. **The headroom is in the numbers.** With the labels' numbers our tracks reach 52–60 (official rules). Reading
   numbers well is worth about 22–30 more points.
4. **Why the reader is weak:**
   - SN-GSR train has only 41 different numbers and few players, so a classifier trained from scratch learns
     appearance.
   - The crops are small and blurred.
   - Many crops show the front or side, where the number is small or hidden.
5. **Next (A7 run 2).** Read numbers with a pretrained scene-text recogniser, which knows digits from millions of
   text images rather than from 41 numbers. Use PARSeq (Apache-2.0), first as is, then fine-tuned on our crops.
   Add a legibility filter that drops crops showing no number, as in the jersey-number literature. Compare at the
   track level with the table above.

Outputs (Drive `outputs/jersey/`):
- `resnet34/` (weights);
- `probs_valid/` and `probs_test/` (per-crop number probabilities);
- `vote_tune.csv`, `gs_hota_a7.csv`, `settings.json`;
- `official_{valid,test}_{pitchlens_a6,pitchlens_a7}.txt`.

## Run 2 (2026-11-25, Colab A100): PARSeq

**Crop.** Only where the number sits: rows 12–50% of the box plus 5% of the width on each side, at the original
resolution, letterboxed to 128 × 32.

**Readers compared on the same tracks.**
1. Run 1's ResNet-34.
2. PARSeq as is (pretrained only).
3. PARSeq fine-tuned for 3 epochs (loss 1.24 → 0.76), only on train crops where the pretrained PARSeq read *some*
   number with confidence ≥ 0.5. These are **1,507 of 101,169 crops (1.5%)**; it was already right on 54% of them.

**Reader alone** (labelled player tracks, valid held-out half, 564 tracks, vote with at least 2 crops):

| Reader | min_conf | Track right | Numbered tracks read right | Wrong number given |
|---|---|---|---|---|
| ResNet-34 (run 1) | 0.7 | 0.220 | 0.200 | 0.617 |
| PARSeq pretrained | 0.3 | 0.170 | 0.056 | 0.142 |
| **PARSeq fine-tuned** | **0.3** | **0.404** | **0.319** | 0.186 |
| PARSeq fine-tuned | 0.5 | 0.390 | 0.290 | 0.062 |

- **The pretrained PARSeq barely reads numbers.** It reads sponsor text ("COMPANY", "SPART") or single strokes. The
  crops are small and blurred, and front views show the sponsor rather than the number.
- **1,500 football crops are enough to change that.** The fine-tuned model reads 1.6× as many numbers as run 1, with
  a third of the wrong numbers.

**Vote** (tuning half, GS-HOTA):

| Reader | Best GS-HOTA | min_conf | min_votes |
|---|---|---|---|
| ResNet-34 | 27.61 | 0.9 | 2 |
| PARSeq pretrained | 25.30 | 0.3 | 1 |
| **PARSeq fine-tuned** | **34.92** | **0.3** | **2** |

**GS-HOTA (our implementation, official rules):**

| Reader | Valid held-out half | All valid | Test |
|---|---|---|---|
| ResNet-34 (run 1) | 28.16 | 27.89 | 29.92 |
| PARSeq pretrained | 27.56 | 26.47 | 26.28 |
| **PARSeq fine-tuned** | **35.96** | **35.46** | **34.83** |

**Official SoccerNet evaluator** (sn-trackeval, all clips of each split, fine-tuned PARSeq):

| | GS-HOTA | DetA | AssA | LocA |
|---|---|---|---|---|
| valid | **36.09** | 20.38 | 63.91 | 92.58 |
| **test** | **35.54** | 20.97 | 60.25 | 92.76 |

Our implementation is again within 0.7 points (test 34.83 against 35.54).

**Official test GS-HOTA by step:**

| Step | Official test GS-HOTA |
|---|---|
| SoccerNet baseline | 22.26 |
| A6, no numbers | 24.00 |
| A7 run 1, ResNet reader | 30.52 |
| **A7 run 2, PARSeq fine-tuned** | **35.54** |

Choi (2026) reports 35.2, but on 11 test clips only.
