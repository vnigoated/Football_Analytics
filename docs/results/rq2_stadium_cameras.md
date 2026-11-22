# RQ2 part 2: how much does the stadium camera matter?

[Experiment 3](exp3_fitted_camera.md) used **one** camera fitted across 10 SkillCorner matches. Real stadiums differ:
the main camera sits 57–97 m from the pitch centre, broadcasters zoom differently (median field of view 18–36°), and
real visibility ranges from 12.9 to 16.5 players per frame. Here every IDSSE and Metrica match is filmed with **each of
20 stadium cameras**, one fitted per SkillCorner match.

Reproduce:

```bash
pitchlens fit-camera --per-match --out configs/cameras/stadiums      # 20 cameras (in-sample, 86–92 % agreement)
pitchlens visibility-bias --match J03WPY --camera-json configs/cameras/stadiums --tag st --sample-every 5 \
    --out outputs/stadiums/idsse                                    # repeat for every match
pitchlens stadium-summary --dir outputs/stadiums/idsse
```

Each fitted stadium camera reproduces its own real visibility within 0.18 players on average. Frames are evaluated
every 0.2 s (`--sample-every 5`; the camera still follows every frame), which changes results by less than 0.1
percentage points (checked on J03WPY).

## Results (95 % intervals, three-level bootstrap: stadium cameras × matches × 60 s blocks)

| | IDSSE (7 matches × 20 cameras) | Metrica (2 matches × 20 cameras) |
|---|---|---|
| Players visible (range over stadiums) | 12.1 – 16.0 | 12.8 – 16.6 |
| **Active pressers, average over stadiums** | **−22.6 % [−25.2, −19.8]** | **−19.0 % [−22.0, −16.1]** |
| Range over stadium cameras | −28.6 % to −13.6 % | −25.2 % to −9.9 % |
| Change per extra visible player | +3.9 pp | +4.2 pp |
| **Pressure on ball carrier, average** | **−1.7 % [−2.2, −1.2]** | **−1.8 % [−2.3, −1.3]** |
| Range over stadium cameras | −3.7 % to −0.5 % | −3.6 % to −0.5 % |

**Which uncertainty matters (IDSSE, active pressers):** stadium camera alone [−24.3, −20.6]; matches + within-match
alone [−24.4, −20.6]; combined [−25.2, −19.8]. Camera choice adds about as much uncertainty as the sample of matches.

## What drives the bias

- **Active pressers:** almost entirely **how many players the broadcast shows** (correlation 0.98 across stadium
  cameras, about 4 percentage points less undercount per extra visible player). Broadcasters that frame the whole
  group of players (15–16 visible) miss about 13 %; tight framing (about 12 visible) misses up to 29 %.
- **Pressure on the ball carrier:** almost entirely **camera lag** (correlation −0.98). 11 of 20 stadium cameras have no
  measurable lag (about −0.8 %); 0.1 s gives about −1.7 %, 0.2 s about −3.1 %, 0.3 s about −3.7 % (IDSSE; Metrica
  the same within 0.3 pp). The single shared camera in Experiment 3 averaged this lag away.

## Conclusions

1. Broadcast tracking undercounts a team's active pressers by about **one fifth on average** (−22.6 % IDSSE,
   −19.0 % Metrica), robust to stadium, match and within-match variation.
2. For **one specific broadcast** the error can be anywhere from about −10 % to −29 %: it is predictable from the
   number of visible players, which can be measured on the broadcast itself. This is the basis for a correction.
3. Pressure on the ball carrier is nearly unbiased with a lag-free camera, but even 0.1–0.3 s of camera lag makes it
   2–5 times worse.

## Limitations

- Stadium cameras are fitted and evaluated on the same match (in-sample); the held-out check is in Experiment 3.
- A-League broadcast styles may differ from Bundesliga or other leagues.
- Each camera's random framing noise is not simulated (it worsened predictions in Experiment 3).
