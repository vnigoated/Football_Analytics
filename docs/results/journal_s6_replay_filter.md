# Journal step 6: main-camera filter on full broadcasts

**Question.** Step 5 showed that only ~64% of a broadcast is the live main camera and that the filter must remove
~97% of the other frames for the correction to stay within 1 pp (if leaked frames show few people). B3's filter
removed no main-camera frames on SN-GSR, but GSR has no replays or close-ups. Does it remove them on real broadcasts?

**Data.** 8 full games from SoccerNet_raw_HQ (`frames-720p-2fps`, under NDA; Colab use approved by the SoccerNet
team; only aggregate statistics reported): 2 EPL, 2 La Liga, Serie A, Bundesliga, UCL, Ligue 1. 86,400 frames,
85,117 inside a labelled shot (SoccerNet-v2 camera labels, label = shot ending at its timestamp). Grass share and cuts
on every frame, RF-DETR people (score >= 0.5, box heights x1.5 to 1080p pixels) on every 2nd frame.
Notebook `colab/12_replay_filter.ipynb` (L4, ~11 min of processing after downloads); per-frame numbers stay private
(`outputs/s6_replays/`, gitignored).

**Alignment.** Agreement of the per-frame cues with the "main camera" labels peaks at offset 0 s (0.906; 0.888 /
0.895 at -/+0.5 s): frame times and labels line up.

## Result 1: the B3 filter is not strict enough

| Camera type (share of frames) | Removed by B3 |
|---|---|
| Live main camera (66%) | 1.2% |
| Close-up (27%) | 61% |
| Behind the goal (3%) | 36% |
| Main-camera replay (3%) | 9% |
| Crowd (1%) | 96% |
| Other (1%) | 34% |

42% of non-main frames kept (16% of the kept frames); mean people on kept frames -1.72 vs live play, so the
correction would move by **-7.1 pp** (per game -3.0 to -11.6; the two Real Madrid broadcasts best). Best B3-style grid
setting: -3.2 pp. Cause: 42% of close-up frames have **no detected person** (the detector was trained on wide views),
and B3 lets frames without people pass the size test.

People count by camera type: live main 15.2 (only 1.8% of frames <= 4 people), close-up 2.3 (84% <= 4).

## Result 2: broadcast filter

`pitchlens.vision.shots.BROADCAST`: grass >= 0.15, median person <= 200 px, **frames without people fail**, decisions
per frame (at 2 fps a shot vote merges close-ups into the main shots around them).

Leave one broadcast out over 24 settings (grass 0.15/0.2/0.3 x box 200/300 x vote x no-people rule; pick the
smallest mean |shift| on 7 games with <= 5% live main-camera frames removed, score on the 8th): **every fold chose
the same setting.**

| | B3 default | Broadcast filter |
|---|---|---|
| Live main camera removed | 1.2% | 1.7% |
| Close-ups removed | 61% | 93% |
| Behind the goal removed | 36% | 62% |
| Main-camera replays removed | 9% | 10% |
| Crowd removed | 96% | 97% |
| Non-main frames kept | 42% | 10% |
| Non-main share of kept frames | 16% | 4.6% |
| People on kept frames vs live play | -1.72 | +0.01 |
| Shift, all 8 games | -7.1 pp | +0.03 pp |
| Held-out shift per game, mean abs (max) | - | **0.30 pp (0.60)** |

Held-out per game (pp): Man Utd-Leicester +0.07, Swansea-Liverpool +0.60, Man City-Barcelona -0.27, PSG-Metz -0.42,
Bayern-Wolfsburg +0.19, Roma-Empoli +0.43, Real Madrid-Valencia -0.27, La Coruna-Real Madrid +0.13. All within 1 pp.

The 10% of non-main frames that still pass are mostly main-camera replays and wide views (behind the goal, other)
whose people counts match live play, so the count does not move. An extra minimum people count (3-8, tested in a first scratch run) was never chosen leaving one game out.

**SN-GSR check (main camera only):** the broadcast filter removes 1.2% of the 80,250 frames (B3 per frame: 0.9%),
mean 1.2% per clip, max 40%, no clip loses most of its frames.

## Caveats

- 8 broadcasts from 2015-2017, one detector, 2 fps; other broadcasters may need re-tuning (same procedure).
- People = detections at score >= 0.5, not the tracked visible-player count; the shift measures the filter's effect
  on the count, not the count's accuracy (step 4).
- Main-camera replays are kept: harmless for the visible-player count, but event metrics must not count them twice.

## Reproduce

```
python -m pitchlens.cli replay-filter --frames outputs/s6_replays/s6_frames.parquet --out outputs/s6_replays/v2
```
