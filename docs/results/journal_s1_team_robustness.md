# Journal step 1: is the pressing bias driven by one team?

**Code:** `pitchlens.analytics.robustness`, `pitchlens team-robustness` · **Run:** 2026-11-25, laptop (seconds) ·
**Input:** paper v1's `outputs/correction/correction_per_broadcast.csv`. This has 360 team-broadcasts: 20 stadium
cameras × 9 matches × 2 pressing teams. The unit is the relative bias of active pressers.

## Concern

In the IDSSE ground truth, Fortuna Düsseldorf plays in 5 of the 7 matches. It is the pressing team in 100 of the
280 IDSSE team-broadcasts.

## Results

### Bias per pressing team

**Every one of the 12 teams is undercounted by 18–28%.**

| Team (IDSSE) | Matches | Mean bias | Range over cameras | Visible players |
|---|---|---|---|---|
| SSV Jahn Regensburg | 1 | −27.9% | −34.7 to −17.6 | 13.49 |
| FC St. Pauli | 1 | −24.6% | −30.3 to −15.5 | 13.74 |
| **Fortuna Düsseldorf** | **5** | **−23.9%** | −34.3 to −9.7 | 13.75 |
| 1. FC Nürnberg | 1 | −22.8% | −28.8 to −12.8 | 14.08 |
| 1. FC Kaiserslautern | 1 | −22.7% | −29.7 to −12.8 | 13.38 |
| Bayer 04 Leverkusen | 1 | −21.9% | −28.5 to −11.9 | 13.77 |
| F.C. Hansa Rostock | 1 | −20.6% | −26.7 to −11.3 | 14.06 |
| 1. FC Köln | 1 | −19.9% | −25.3 to −11.8 | 13.87 |
| VfL Bochum 1848 | 1 | −18.5% | −23.3 to −9.9 | 13.77 |
| FC Bayern München | 1 | −17.8% | −23.4 to −9.8 | 13.87 |

The two Metrica teams, Home and Away, have −19.5% and −18.4%.

### Team-balanced mean

When every team counts equally, the IDSSE mean is **−22.1% [95% CI −23.9, −20.3]**. The interval comes from
resampling teams. Weighting every team-broadcast equally gives −22.6%. Fortuna's extra weight therefore moves the
mean by 0.5 percentage points (pp).

### Leave one team out

Each team is dropped together with every match it plays in:

| Left out | Matches left | Mean bias | Slope (pp per visible player) |
|---|---|---|---|
| none | 7 | −22.6% | 4.05 |
| each other team, one at a time | 6 | −21.9% to −23.2% | 3.93 to 4.17 |
| **Fortuna Düsseldorf** | **2** | **−19.5%** | **3.90** |

Without Fortuna, two IDSSE matches remain: Köln–Bayern and Leverkusen–Bochum. Their −19.5% matches the independent
Metrica replication (−19.0%). The slope against the number of visible players stays at about 4 pp per player in
every case.

### Team effect at equal visibility

A least-squares fit of bias on visible players plus a Fortuna indicator gives Fortuna an extra
**−1.9 pp [95% CI −5.0, +1.7]**. The interval comes from resampling matches. The difference is not significant:
Fortuna's broadcasts are biased like the others once visibility is accounted for.

### The correction with the team unseen

Each row's correction line is fitted without its stadium camera **and** without every match of its pressing team:

| Evaluation | Mean error | Mean abs. error | 95th pct abs. |
|---|---|---|---|
| Paper v1: camera and match left out | +0.0% | 3.05% | 7.29% |
| **Camera and all matches of the team left out** | −0.5% | **3.35%** | 7.96% |

## Conclusion for the paper

The visibility bias is not a property of one team:
- every team is undercounted by 18–28%;
- weighting teams equally changes the mean by only 0.5 pp;
- leaving out any team, including Fortuna with its 5 matches, keeps the bias between −19.5% and −23.2% and the slope
  at about 4 pp per player;
- Fortuna shows no significant team effect at equal visibility;
- the correction still reaches a 3.35% mean absolute error when the pressing team has never been seen.

Outputs (laptop `outputs/robustness/`): `team_robustness.json`, `leave_one_team_out_idsse.csv`,
`correction_team_unseen.csv`.
