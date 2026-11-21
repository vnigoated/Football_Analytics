# PitchLens: Research Plan

> Companion to [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md). The overview explains *what we build*; this document explains *what we want to find out* and *how we will prove it*.
>
> **Revised 2026-11-20 after a literature and licence check.** Several ideas from the first draft have already been published (see §2 and §7). The plan now focuses on the gaps that are still open.

---

## 1. The core research problem

Broadcast footage is the cheapest and most widely available source of football data. But a TV camera **only shows part of the pitch**: on average about **14 of 22 players** are visible in a frame (SkillCorner open data notes).

Most football metrics were designed for **full tracking data**, where all 22 players are known at all times.

### What is already known (literature check)

| Topic | Status | Key work |
|---|---|---|
| Missing players bias **pitch control** | **Already published** | arXiv 2607.11548 (2026): simulated broadcast camera on Metrica data; about 25–27 percentage points of hidden-zone pitch-control error without imputation |
| Missing players bias **physical metrics** (distance, speed) | **Already published** | "Broadcast analytics", *Science and Medicine in Football* (2025): physical metrics hit hard, formation recognition robust |
| **Imputing off-screen players** | **Well studied** | Graph Imputer (*Scientific Reports* 2022); Agent Imputer (AAMAS 2023); continuous tracking from broadcast (*Royal Society Open Science* 2025: 3.45 m mean error overall, 7.33 m for off-camera players); training-free formation anchor (arXiv 2607.11548); FOOTPASS; TacSIm |
| **Event detection** from broadcast tracking | **Already published** | *Sports Engineering* (2026); PathCRF (arXiv 2602.12080); arXiv 2202.00804 |
| Broadcast bias on **pressing intensity** | **No study found** | Only Bekkers' pressing intensity (2025), defined on full tracking |
| **Propagating calibration and detection uncertainty** into tactical metrics | **No football-specific study found** | Closest: block-bootstrap intervals in 2607.11548 (temporal sampling only); FIFA's quality programme for broadcast tracking |

### Revised main question

> *How reliable is **pressing intensity** when computed from broadcast video, and can we report it with **honest, calibrated error bars** that account for every stage of the pipeline: camera calibration, detection, tracking and off-screen imputation?*

---

## 2. Research questions and hypotheses

### RQ1 (main): Visibility bias in pressing metrics

**Question:** When players are off-screen, how much do **pressing intensity** and related out-of-possession metrics drift from their true values? Examples are pressure on the ball carrier, passing-lane coverage and PPDA-style counts.

> Prior work covers pitch control and physical metrics. **Pressing is not covered.** That is our gap.

**Hypotheses:**
- **H1a:** Pressing intensity is **underestimated** in build-up from deep areas, because the camera follows the ball and cuts out distant pressers and cover players.
- **H1b:** **Passing-lane coverage** suffers more bias than **pressure on the ball carrier**, because the nearest presser is usually on screen.
- **H1c:** Team- and player-level **rankings** stay fairly stable (high rank correlation) even when absolute values are biased.

### RQ2 (main): End-to-end uncertainty propagation

**Question:** Can we pass uncertainty from **every pipeline stage** (homography, detection, tracking, imputation) into the final metric and produce **well-calibrated** intervals?

> The 2607.11548 preprint reports bootstrap intervals for **temporal sampling only**. We found no football work that propagates **calibration and detection error** through to tactical metrics.

**Hypotheses:**
- **H2a:** Monte Carlo sampling over homography perturbations (from keypoint reprojection error) and imputation samples produces intervals with roughly nominal coverage. For example, about 90% of true values fall inside the 90% intervals.
- **H2b:** **Calibration error dominates** uncertainty in wide shots far from the camera. **Imputation error dominates** in zoomed-in shots.

### RQ3 (supporting): Metric-aware imputation for pressing

**Question:** Do existing imputation methods help pressing metrics as much as they help pitch control? Does tuning imputation for the **downstream metric** beat tuning for position error?

> This is **not a headline contribution**. Imputation is well studied. We **reuse and compare** existing approaches (formation anchor, constant velocity, a Graph Imputer-style model) and look only at their effect on pressing.

**Hypothesis:**
- **H3:** Lower position error does **not always** mean lower pressing error. Velocity and heading errors matter more for pressing than for pitch control.

### Also included: replication and extension studies

These four topics **are already published**, but we still do all of them. They're needed for a complete pipeline, and independently reproducing a result on new data is a valued research contribution. We present them **honestly as replications and extensions**, always citing the original work, and never as our novelty.

| Topic | Original work | What we add |
|---|---|---|
| **R1. Hidden players skew pitch control** | arXiv 2607.11548 (Metrica) | Reproduce on Metrica, then **extend to IDSSE** (new league, 25 fps optical tracking) and to a SkillCorner-fitted camera. Do the 25–27 pp errors hold? |
| **R2. Hidden players skew physical metrics** | *Science and Medicine in Football* (2025) | Re-measure distance, high-speed running and sprint counts under the same virtual camera, so all metric families are compared **in one framework**: physical vs pitch control vs pressing |
| **R3. Off-screen player imputation** | Graph Imputer (2022), Agent Imputer (2023), RSOS (2025), 2607.11548, FOOTPASS, TacSIm | Re-implement or reuse several methods and benchmark them **side by side on the same data**, judged by position error **and** by downstream metric error (feeds RQ3) |
| **R4. Event detection from tracking** | *Sports Engineering* (2026), PathCRF (2026), arXiv 2202.00804 | Build it (xT needs it), evaluate on **real broadcast-derived tracking** from our own pipeline, and measure how detection errors change **xT** values |

**Why this strengthens the project:**
- **Credibility:** if we reproduce known results first, our new pressing and uncertainty findings are easier to trust.
- **One unified benchmark:** nobody has compared pitch control, physical, pressing and xT metrics under the **same** broadcast simulation and the **same** imputation methods.
- **Open-source value:** the repo becomes a single place where all of these methods run on the same data.

---

## 3. Key method: simulate the broadcast camera on full data

```
 Full tracking data (all 22 players, known truth)
          │
          ▼
 Virtual broadcast camera
 (moving window that follows the ball; visibility statistics
  fitted to SkillCorner real broadcast data)
          │
          ▼
 "Broadcast-like" data (only players inside the window)
          │
          ├──► compute pressing ──► compare with full-data pressing  (RQ1)
          ├──► add calibration + detection noise + imputation
          │        └──► Monte Carlo ──► test interval coverage        (RQ2)
          └──► impute missing ──► pressing error per method           (RQ3)
```

**Positioning vs prior work:** 2607.11548 uses a similar simulation on Metrica data for pitch control. We **adopt and cite** their protocol, compare against it directly, and extend it in three ways:
1. to **pressing metrics**;
2. to a **second, larger dataset** (IDSSE, 25 fps, 7 matches);
3. with a camera model **fitted to real broadcast visibility** (SkillCorner).

We then **check the results on real broadcast video** (SoccerNet-GSR) using our full computer-vision pipeline.

---

## 4. Datasets (licences checked 2026-11-20)

| Dataset | Contents | Licence / terms | Used for |
|---|---|---|---|
| **IDSSE / Sportec open data**: Bassek, Rein, Weber & Memmert, *Scientific Data* (2025), DOI 10.1038/s41597-025-04505-y | 7 matches (2 Bundesliga, 5 2. Bundesliga, 2022/23); TRACAB optical tracking at 25 fps + official events. Mirror: Hugging Face `pysport/idsse-data`; loads with kloppy | **CC-BY 4.0**: free use including commercial, **attribution required** | **Primary ground truth** for RQ1–RQ3 |
| **Metrica Sports sample data** (`metrica-sports/sample-data`) | 3 anonymised games (2 CSV, 1 EPTS/JSON) with synchronised events. Full-pitch coordinates (105×68 m), 25 fps. **Checked (Nov 2026, kloppy):** all 22 players present in every frame of the first 20,000 frames of all 3 games (game 3 lists 35 players incl. substitutes) | **No licence file.** README only says *"Please be responsible with the use of this data"* and *"If you use it for anything public, please aknowledge the source."* | Secondary test set and **comparison with 2607.11548** (same data). Research use with acknowledgement; **don't redistribute; don't assume commercial rights** |
| **SkillCorner open data** (A-League 2024/25) | 10 matches of real broadcast tracking at 10 fps + dynamic events | MIT (per repo); credit SkillCorner. Re-check the LICENSE file before release | Fit the **virtual camera** visibility model |
| **SoccerNet-GSR** (`SoccerNet/SN-GSR-2025` on Hugging Face) | Broadcast clips + ground-truth minimap positions; train 9.8 GB, valid 11.2 GB, test 8.9 GB, challenge 5.3 GB (≈35 GB total) | **Not gated** (checked 2026-11-20); dataset card says **GPL-3.0**. Raw full-match videos (`SoccerNet_raw_HQ`) and ball-action sets **are** NDA-gated, but we don't need them | End-to-end validation on real video; GS-HOTA |
| **SoccerNet-Calibration / Tracking / ReID / Jersey** (HF, not gated) | Pitch line annotations, tracking clips, re-ID crops, jersey numbers | Check each dataset card's licence | Training the keypoint, detection, re-ID and jersey models |
| **StatsBomb open data** | Large event dataset | StatsBomb open-data user agreement (attribution; check before use) | Training the xT grid |

**Data to-do:**
- [x] Load Metrica games in kloppy and confirm all 22 players + ball are present in every frame before using them as ground truth. (Done: 22/22 players in 100% of checked frames, games 1–3.)
- [ ] Read the SkillCorner and StatsBomb licence files once more before the public release.

---

## 5. Experiments

### Experiment 1: Visibility bias across all metric families (RQ1 + R1 + R2)

1. Fit the virtual camera to SkillCorner: field-of-view width, how it follows the ball, zoom changes.
2. **R1 replication:** reproduce the 2607.11548 pitch-control result on Metrica with their camera settings.
3. Apply the camera to all IDSSE and Metrica matches.
4. Compute every metric family on **full** data and on **masked** data:
   - **Pitch control** (R1 extension)
   - **Physical:** total distance, high-speed running, sprints (R2 replication)
   - **Pressing (novel):** pressing intensity (`unravelsports`), pressure on the ball carrier, passing-lane coverage, PPDA-style counts
   - **xT** on true events
5. Measure the gap, and **rank the metric families** by how fragile they are under broadcast visibility.

**Metrics:**
- Frame-level: MAE, bias (mean signed error)
- Aggregate: error in team and player per-match totals
- Rankings: Spearman rank correlation

**Breakdowns:** pitch zone, game phase, number of hidden players, and camera zoom level.

### Experiment 2: Uncertainty propagation (RQ2)

1. Model the error sources:
   - **Calibration:** perturb the homography using keypoint reprojection error, measured on SoccerNet-GSR
   - **Detection:** random misses and position jitter, scaled by confidence
   - **Tracking:** ID swaps at the measured rate
   - **Imputation:** samples from the imputation model's predicted distribution
2. Run N Monte Carlo samples through the pressing computation.
3. Use block bootstrap over time (as in 2607.11548) for temporal sampling uncertainty.

**Metrics:**
- Interval **coverage** at 50/80/90/95% levels
- Interval width
- Variance breakdown: which stage contributes most of the uncertainty?

### Experiment 3: Imputation benchmark (RQ3 + R3)

**Compare (reusing published methods where code exists):**
1. No imputation (drop hidden players)
2. Hold last seen position
3. Constant velocity
4. Formation anchor (as in 2607.11548)
5. Graph Imputer-style learned model

**Protocol:** train on IDSSE, test on Metrica.

**Metrics:**
- Position error (ADE) and velocity/heading error, compared with the published numbers (e.g. RSOS 2025: 7.33 m for off-camera players)
- Error after imputation for **every metric family** from Experiment 1. **Pressing error** is the main outcome.

### Experiment 4: End-to-end on real broadcast video

1. Run the full PitchLens computer-vision pipeline on SoccerNet-GSR.
2. Report **GS-HOTA**.
3. Compute pressing intensity on our predicted positions vs the ground-truth minimap.
4. Check that the **real** errors fall inside the **predicted** uncertainty intervals from Experiment 2. This is the key test that the error bars carry over to real video.

### Experiment 5: Event detection and xT (R4)

1. Detect passes and carries from tracking (a rule-based possession model, then a learned classifier), with PathCRF-style methods as baselines.
2. Evaluate on three inputs:
   - **full** IDSSE tracking
   - **broadcast-masked** tracking
   - **real broadcast tracking** from our pipeline (SoccerNet-GSR)
3. Compute xT from detected events vs true events.

**Metrics:**
- Precision/recall/F1 at ±1 s, compared with published numbers
- xT correlation and player-ranking stability (Spearman)

### Ablations
- Replay/close-up filtering on vs off
- Keypoint-only homography vs PnLCalib refinement
- Temporal smoothing of calibration on vs off
- Detector choice (RF-DETR vs YOLO), judged by the effect on downstream metrics, not just mAP

---

## 6. Expected contributions (revised)

1. **First measurement of broadcast visibility bias on pressing metrics.** No prior study was found; prior work covers pitch control and physical metrics.
2. **End-to-end uncertainty propagation** from camera calibration to tactical metric, with calibrated intervals validated on real broadcast video. No football-specific prior work was found.
3. **A comparison of imputation methods judged by pressing accuracy** rather than position error, reusing existing methods.
4. **An open, reproducible pipeline** from broadcast video to standard tracking data, with published benchmarks.

**Replication and extension contributions (clearly labelled as such):**

5. **Independent replication** of broadcast bias in pitch control (2607.11548) and physical metrics (2025), **extended to a new dataset** (IDSSE).
6. **A unified benchmark:** pitch control, physical, pressing and xT metrics, plus several imputation and event-detection methods, all under one broadcast simulation on the same data.

> Novelty is based on searches run on 2026-11-20. Re-run the searches before submitting. The field is moving fast: 2607.11548 appeared only in July 2026.

---

## 7. Related work (checked)

| Area | Work |
|---|---|
| Expected Threat | Karun Singh, "Introducing Expected Threat (xT)" (2019 blog) |
| Action valuation | Decroos et al., VAEP (KDD 2019); Van Roy et al., xT vs VAEP comparison; `socceraction` (maintained for reproducibility only) |
| Pitch control | Spearman, "Beyond Expected Goals" (MIT Sloan 2018) |
| Pressing | Bekkers, "Pressing Intensity" (arXiv 2501.04712, v2 June 2025); `unravelsports` |
| **Broadcast visibility bias** | "Training-Free Off-Screen Player Imputation for Broadcast-Based Spatial Football Analytics" (arXiv 2607.11548, 2026), **the closest prior work**; "Broadcast analytics", *Science and Medicine in Football* (2025, doi 10.1080/24733938.2025.2533808); FIFA blog on broadcast tracking data quality |
| **Off-screen imputation** | "Multiagent off-screen behavior prediction in football" (Graph Imputer), *Scientific Reports* (2022); "Inferring Player Location in Sports Matches" (Agent Imputer), AAMAS 2023; "Continuous football player tracking from discrete broadcast data", *Royal Society Open Science* (2025); FOOTPASS (arXiv 2511.16183); TacSIm (arXiv 2603.25199) |
| Event detection | "Automatic event detection in association football using broadcast-derived tracking data", *Sports Engineering* (2026); PathCRF (arXiv 2602.12080); arXiv 2202.00804 |
| Broadcast tracking / GSR | Somers et al., SoccerNet-GSR (CVPRW 2024); SoccerNet 2024/2025 challenge reports; "From Broadcast to Minimap" (arXiv 2504.06357) |
| Camera calibration | PnLCalib (arXiv 2404.08401); TVCalib (arXiv 2207.11709) |
| Foundation models | SoccerMaster (arXiv 2512.11016) |
| Datasets | Bassek et al., *Scientific Data* (2025) |

**To do:** read 2607.11548 in full first. It is the paper we must position against, and its code or protocol may be reusable.

---

## 8. Timeline (runs alongside the engineering roadmap)

| Weeks | Research work |
|---|---|
| 1–2 | Read 2607.11548, Graph Imputer and the 2025 *Royal Society Open Science* paper in detail; load IDSSE, Metrica and SkillCorner via kloppy; check Metrica completeness |
| 3–4 | Build and fit the virtual camera; **R1** reproduce 2607.11548's pitch-control result |
| 5–6 | **Experiment 1** (all metric families incl. **R2** physical + novel pressing) |
| 7–8 | **Experiment 3** (imputation benchmark, **R3**) on Colab A100 |
| 9 | **Experiment 2** (uncertainty propagation) + **Experiment 5** (event detection, **R4**) |
| 10 | **Experiment 4** (end-to-end on SoccerNet-GSR) |
| 11 | Paper draft |
| 12 | Re-run the novelty search, polish, release code + results, submit preprint |

---

## 9. Risks and backup plans

| Risk | Backup plan |
|---|---|
| Someone publishes pressing bias first | Lead with uncertainty propagation (RQ2), which is the more distinctive contribution |
| Can't reproduce 2607.11548's numbers | Report the difference openly; that is useful to the field |
| Virtual camera not realistic enough | Validate against SkillCorner visibility statistics; check with Experiment 4 on real video |
| Metrica frames incomplete | Use IDSSE only as ground truth (CC-BY 4.0, larger) |
| Only 7 + 3 full-tracking matches | Data augmentation (pitch flips, time windows); block bootstrap; report uncertainty honestly |
| SoccerNet dataset moved or gated later | Pin the download revision; Experiments 1–3 don't need video |
| Colab session limits | Resumable training, checkpoints every epoch |

---

## 10. Where to publish

| Venue | Fit |
|---|---|
| **ECML-PKDD MLSA workshop** | Pressing bias + uncertainty (main fit) |
| **CVPR CVsports workshop** | Pipeline + end-to-end uncertainty on video |
| **MIT Sloan Sports Analytics Conference** (research papers) | Practical impact: "how much can you trust broadcast pressing data?" |
| **StatsBomb Conference** | Analytics audience |
| ***Journal of Sports Analytics* / *Science and Medicine in Football*** | Longer version; the latter published the 2025 "Broadcast analytics" study |
| **arXiv** | Preprint linked from the repo |

---

## 11. Reproducibility and licence checklist

- [ ] All code public; configs for every experiment in the repo
- [ ] Fixed random seeds; exact package versions pinned
- [ ] Scripts that rebuild every table and figure from raw data
- [ ] **Credit IDSSE (CC-BY 4.0)**, Metrica, SkillCorner and StatsBomb as each requires
- [ ] Don't redistribute Metrica, SoccerNet or broadcast video; provide download scripts instead
- [ ] Model weights published only where dataset terms allow
- [ ] Results table in the README, updated with each release

---

## 12. Proposed paper title (draft)

> **"How Much Can You Trust Broadcast Pressing Data? Visibility Bias and End-to-End Uncertainty in Football Tracking Metrics"**
