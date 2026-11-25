# Journal step 2: a tenth ground-truth match (Metrica Sample Game 3)

**Code:** `pitchlens.data.metrica` (game 3: FIFA EPTS tracking via kloppy 3.19.1, JSON events via
`events_from_json`) · **Run:** 2026-11-25, laptop

```bash
pitchlens pressing-baseline --match metrica-3 --out outputs/metrica/pressing_baseline --no-plot   # 155 s
pitchlens visibility-bias --match metrica-3 --camera-json configs/cameras/stadiums --tag st --sample-every 5 \
    --gt-dir outputs/metrica/pressing_baseline --out outputs/stadiums/metrica                    # 280 s
pitchlens stadium-summary --dir outputs/stadiums/metrica
pitchlens bias-correction                       # now 400 team-broadcasts (20 cameras × 10 matches × 2 teams)
pitchlens team-robustness --out outputs/robustness_10
pitchlens count-error-propagation --table outputs/correction/correction_per_broadcast.csv --out outputs/b2_propagation_10 --no-plot
```

The 9-match results are kept in `outputs/correction_9matches/` and
`outputs/stadiums/metrica/stadium_per_camera_2games.csv`.

## Data

Game 3 has 104,479 ball-in-play frames with an owner, about 70 minutes, with all 22 players tracked. Possession is
rebuilt from the JSON events with the same rules as games 1–2: Team A owns 53% and Team B 47%. Of these frames,
7,131 (6.8%) have no ball carrier in the possessing team and are dropped, leaving 97,348. That share is higher than
for IDSSE (0.06–0.2%) because Metrica's possession comes from hand-annotated events. The two teams are anonymised
("Team A" and "Team B").

## Results

### Game 3

Over the 20 stadium cameras, game 3 shows 14.20 visible players per frame and an active-presser bias of
**−22.5%** (Team A −23.5%, Team B −21.6%). The correction's mean absolute error on this match is 2.5%.

### Metrica replication, 2 → 3 matches

| Target | 2 matches | **3 matches** (combined 95% interval) |
|---|---|---|
| Active pressers | −19.0% [−22.0, −16.1] | **−20.2% [−23.2, −17.1]** |
| Pressure on the ball carrier | −1.8% [−2.3, −1.3] | **−1.8% [−2.3, −1.3]** |
| Slope (pp per visible player) | 4.2 | **4.2** |

### Bias per match (all 10)

| Match | Bias | Visible players | Corrected MAE |
|---|---|---|---|
| IDSSE J03WMX | −18.8% | 13.87 | 4.9% |
| IDSSE J03WN1 | −20.2% | 13.77 | 3.7% |
| IDSSE J03WOH | −26.5% | 13.49 | 4.1% |
| IDSSE J03WOY | −21.8% | 14.06 | 1.7% |
| IDSSE J03WPY | −23.1% | 14.08 | 2.5% |
| IDSSE J03WQQ | −21.9% | 13.74 | 3.5% |
| IDSSE J03WR9 | −25.7% | 13.38 | 3.8% |
| Metrica 1 | −20.5% | 14.09 | 1.6% |
| Metrica 2 | −17.5% | 14.82 | 1.5% |
| **Metrica 3** | **−22.5%** | 14.20 | 2.5% |

### Correction, 9 → 10 matches

Leaving out the stadium camera and the match:

| | 9 matches (360) | **10 matches (400)** |
|---|---|---|
| Raw mean abs. error | 21.8% | 21.9% |
| Corrected mean abs. error | 3.05% | **3.00%** |
| Corrected 95th pct abs. | 7.29% | **6.95%** |
| Pooled slope | 4.14 pp per player | 4.12 pp per player |
| With the pressing team also unseen (step 1) | 3.35% (p95 7.96%) | **3.18% (p95 7.57%)** |

### Team checks with 14 teams (step 1 repeated)

- The 14 teams are undercounted by 17.8–27.9%. The new teams are Team A at −23.5% and Team B at −21.6%.
- The team-balanced Metrica mean is −20.8% [−22.5, −19.0].
- IDSSE is unchanged: −22.1% [−23.8, −20.4].
- Fortuna's extra bias at equal visibility is −1.9 pp [−5.3, +1.5].

### B2 with the 10-match table

| | 9 matches | 10 matches |
|---|---|---|
| True count, mean abs. error | 3.05% | **3.00%** |
| True count, p95 | 7.29% | **6.95%** |
| Calibrated video count, whole match, mean abs. error | 3.09% | **3.02%** |
| Calibrated video count, whole match, p95 | 7.27% | **7.10%** |
| Calibrated video count, single clip, mean abs. error | 4.67% | 4.64% |
| Calibrated video count, single clip, p95 | 12.63% | 12.56% |

## Conclusion

The tenth match behaves like the other nine: a −22.5% bias at 14.2 visible players, on the same 4 pp per player
line. Adding it narrows the corrected error slightly (3.00%, p95 6.95%) and leaves every earlier conclusion
unchanged. The ground truth is now 10 matches with 14 different teams. The one-team concentration (Fortuna in 5 of 7
IDSSE matches) remains, but step 1 shows it does not drive the result.
