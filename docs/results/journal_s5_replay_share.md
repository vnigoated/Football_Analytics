# Journal step 5: how much of a broadcast is not the live main camera, and how much leak the count tolerates

**Code:** `pitchlens.analytics.broadcast_shots`, `pitchlens replay-share` · **Data:** SoccerNet-v2 camera-shot
labels (Hugging Face `SoccerNet/SN-Labels`, `Labels-cameras.json`; not redistributed) · **Run:** 2026-11-26, laptop
(about 2 min)

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('SoccerNet/SN-Labels', repo_type='dataset', allow_patterns=['*Labels-cameras.json','*Labels-v2.json'], local_dir='data/SN-Labels')"
pitchlens replay-share            # -> outputs/s5_replays/
```

## Data and reading the labels

- **Coverage.** 500 games (2014–2017; EPL, La Liga, Serie A, Bundesliga, Ligue 1, Champions League) have camera labels.
  Only **196** are labelled throughout: 300 or more changes and no gap over 3 minutes. The rest are labelled only
  around some moments, with gaps of 5–6 minutes, and are left out. The 196 games are 297 hours of broadcast.
- **Which shot a label describes.** Each label describes the shot that **ends** at its timestamp, not the one
  starting there. We checked this against the action labels of 203 games:

  | Action | Live main camera, label = shot ending | Label = shot starting |
  |---|---|---|
  | Shots on target | 99% | 2% |
  | Goals | 99% | 1% |
  | Fouls | 93% | 8% |
  | Kick-offs | 89% | 4% |
  | Substitutions | 3% (close-ups) | — |

  So a shot runs from the previous change to its own label, and the first label of each half is dropped.

## Results

### Broadcast time by camera (196 games)

| Camera | Live | Replay | Other / logo |
|---|---|---|---|
| Main camera (centre, left, right) | **64.3%** | 1.9% | 0.1% |
| Close-ups (player/referee, staff, corner) | 24.1% | 3.9% | 1.0% |
| Behind the goal | 0.7% | 1.8% | 0.1% |
| Crowd | 0.8% | 0.0% | 0.1% |
| Other (spider cam, inside the goal, goal-line, other) | 0.4% | 0.5% | 0.3% |

Per game:

| Share of the broadcast | Mean | Range |
|---|---|---|
| Live main camera | 64% | 50–74% |
| Replays | 8% | 3–17% |
| Main-camera replays | 1.9% | 0.5–5.1% |
| Close-ups | 29% | 21–41% |

A median non-main shot lasts 4.0 s.

The competitions are similar. Live main camera ranges from 57% (Ligue 1, only 5 games) to 67% (Serie A), and
replays from 7% (Serie A, Bundesliga) to 10% (La Liga).

### Leak tolerance of the visible-player count

A broadcast's visible-player count averages the frames kept by the main-camera filter (B3). If a fraction of the
non-main frames leaks through, those frames pull the average down. They show fewer players, so the corrected
pressing value moves by 4.1 pp per player of shift. The table uses a typical game (35% non-main) and a busy game
(the 90th percentile: 42% non-main), with 14 visible players on live main-camera frames.

| Leaked frames show | Shift at 10% leak, typical / busy game | Largest leak for ≤ 1 pp, typical / busy game |
|---|---|---|
| 0 players (close-up) | −3.0 / −3.8 pp | 3.3% / 2.5% |
| 3 players (close-up) | −2.3 / −3.0 pp | 4.2% / 3.2% |
| 8 players (wide non-main shot) | −1.3 / −1.6 pp | 7.8% / 6.0% |
| As live (main-camera replay) | 0 | any |

## Conclusion

- **Main-camera replays don't matter for the count.** They are 1.9% of a broadcast (at most 5.1%). They look like
  live play, so no appearance filter can remove them, but they show players the way live play does, so they don't
  bias the count.
- **Close-ups do matter: about 29% of the broadcast.** The filter must remove **at least about 97%** of non-main
  frames to keep the correction within 1 pp. That is the requirement step 6 tests on real full broadcasts.
- **Close-ups are the easy case for B3's cues.** People are larger than 300 px tall and there is little grass. A
  frame without a pitch homography (no pitch lines found) is a second, independent check.
- **Without any filter the correction would be off by 9–24 pp,** so the filter is essential for real video. That
  belongs in the paper's method and limitations.
- **Caveat:** the labelled games are 2014–2017 broadcasts. Modern broadcasts may differ, and step 6 uses the same
  era.
