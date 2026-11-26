# PitchLens: Football Analytics from Broadcast Video

[![CI](https://github.com/vnigoated/Football_Analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/vnigoated/Football_Analytics/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%E2%80%933.12-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24557591.svg)](https://doi.org/10.5281/zenodo.24557591)

**PitchLens** turns ordinary TV footage of a football match into **tracking data** (where every player and the ball is, in metres). It then computes **pressing intensity** and **Expected Threat (xT)**, each reported **with honest error bars**.

> **Status: v1.0.0.** This release accompanies the journal manuscript on visibility bias in broadcast pressing
> metrics. The analytics core, the broadcast-camera simulation and the video pipeline work; possession, events and a
> dashboard are next (see roadmap).

- 📖 **Project in plain words:** [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)
- 🔬 **Research plan:** [docs/RESEARCH.md](docs/RESEARCH.md)
- 🗺️ **Phase 2 plan (video pipeline):** [docs/PHASE2_PLAN.md](docs/PHASE2_PLAN.md)

---

## Why

Clubs pay a lot for tracking data. Broadcast video is free, but a TV camera shows only about **14 of 22 players** at a time. PitchLens builds an open broadcast→tracking pipeline **and measures how much that missing view distorts tactical metrics**, with a focus on **pressing**, which hasn't been studied in this setting yet.

## Pipeline

```
TV video → detect players & ball → track → split teams → pitch keypoints
        → homography (screen → metres) → clean + skip replays
        → events (passes, carries) → pressing intensity, xT (± uncertainty)
```

## What works today

| Module | Status |
|---|---|
| `pitchlens.data` | ✅ IDSSE open tracking/events (via kloppy), Metrica, SkillCorner; SoccerNet-GSR and SportsMOT readers |
| `pitchlens.analytics.pressing` | ✅ Pressing intensity (Bekkers 2025, via `unravelsports`) + per-frame summaries |
| `pitchlens.analytics.xt` | ✅ Expected Threat grid (Singh 2019), fit + rate actions |
| `pitchlens.analytics` (bias) | ✅ Correction from the number of visible players, robustness, count-error propagation, broadcast composition |
| `pitchlens.geometry` | ✅ Pitch landmark model, RANSAC homography with reprojection error |
| `pitchlens.simulation` | ✅ Virtual broadcast camera fitted to real broadcasts (SkillCorner, SoccerNet-GSR), player hiding, bias experiments with confidence intervals |
| `pitchlens.vision` | ✅ Detection (RF-DETR), tracking, teams, pitch keypoints, jersey numbers, main-camera filter |

## Quickstart

```bash
git clone https://github.com/vnigoated/Football_Analytics.git
cd Football_Analytics
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

**Ground-truth pressing baseline** on full IDSSE tracking (no GPU needed; the data downloads automatically):

```bash
# quick test on the first 3,000 frames
pitchlens pressing-baseline --match J03WPY --limit 3000
# full match
pitchlens pressing-baseline --match J03WPY
```

Outputs go to `outputs/pressing_baseline/`: a Parquet file of per-frame pressing and a timeline plot.

**Download SoccerNet-GSR** (≈21 GB for train + valid; best done on Colab):

```bash
pitchlens download-gsr --splits train valid --dest data/SN-GSR-2025
```

> Use Python **3.11 or 3.12**. `unravelsports` needs 3.11+, and some dependencies don't support 3.13+ yet. On Windows: `py -3.11 -m venv .venv`.

## Training on Colab (A100)

See [colab/00_setup.ipynb](colab/00_setup.ipynb). It clones the repo, installs it, downloads data to `/content` (fast local SSD) and saves checkpoints to Google Drive.

## Project layout

```
src/pitchlens/
  data/         dataset loaders and downloaders
  geometry/     pitch model, homography
  analytics/    pressing intensity, xT, bias correction, broadcast composition
  simulation/   virtual broadcast camera (research core)
  vision/       detection, tracking, teams, keypoints, jersey numbers, main-camera filter
  cli.py        `pitchlens` command
tests/          unit tests (no network needed)
configs/        fitted broadcast cameras (A-League, SoccerNet-GSR)
docs/           project overview, research plan, one write-up per experiment (docs/results/)
colab/          Colab notebooks (GPU steps)
paper/          manuscripts and figure scripts (paper/journal/ = journal version)
```

## Reproducing the paper

Every number in the journal manuscript ([paper/journal/main.tex](paper/journal/main.tex)) comes from a `pitchlens`
command or a Colab notebook. Each step has a write-up in [docs/results/](docs/results/) with its exact command,
settings and outputs; [docs/PAPER_V2_NOTES.md](docs/PAPER_V2_NOTES.md) maps every result to its source.

| Paper section | How to reproduce | Write-up |
|---|---|---|
| Camera fit and validation | `pitchlens fit-camera` | `exp3_fitted_camera.md`, `rq2_stadium_cameras.md` |
| Bias of pressing metrics, drivers | `pitchlens visibility-bias`, `visibility-summary`, `visibility-ci`, `stadium-summary` | `exp1*`, `exp2*`, `rq2_*` |
| Correction (10 matches) | `pitchlens bias-correction` | `journal_s2_metrica_game3.md` |
| Robustness to teams | `pitchlens team-robustness` | `journal_s1_team_robustness.md` |
| Second league's cameras | `pitchlens fit-camera-gsr`, `cross-league` | `journal_s3_camera_gsr.md` |
| Real video: game state, count | notebooks 06 and 08, `pitchlens count-error-propagation` | `phase2_a6_pipeline.md`, `phase2_b1/b2_*` |
| Count on an independent dataset | notebook 11 (SportsMOT) | `journal_s4_sportsmot_count.md` |
| Broadcast composition | `pitchlens replay-share` | `journal_s5_replay_share.md` |
| Main-camera filter on full broadcasts | notebook 12, `pitchlens replay-filter` | `journal_s6_replay_filter.md` |
| Figures | `python paper/journal/make_journal_figures.py` | [paper/journal/README.md](paper/journal/README.md) |

Data are not included and must be downloaded from their sources (see below). The full SoccerNet broadcast videos used
in the last step are available from the SoccerNet team under a non-disclosure agreement; no frames or per-frame
measurements from them are in this repository.

## Roadmap

| Status | Milestone |
|---|---|
| ✅ | Analytics core, IDSSE pressing baseline, broadcast-camera simulation, visibility-bias study and correction |
| ✅ | Video pipeline: detection, tracking, teams, keypoints and homography, jersey numbers, main-camera filter; end-to-end on SoccerNet-GSR (GS-HOTA) |
| ✅ | v1.0.0 with the journal manuscript |
| Next | Pressing report on a broadcast with the correction applied |
| Next | Ball possession, pass and carry detection; xT on detected events |
| Next | Minimap and dashboard; kloppy export |

## Data and licences

Datasets are **never committed** to this repo; download scripts are provided instead.

| Data | Licence | Credit |
|---|---|---|
| IDSSE / Sportec open data | CC-BY 4.0 | Bassek, Rein, Weber & Memmert, *Scientific Data* (2025), doi:10.1038/s41597-025-04505-y |
| SoccerNet SN-GSR-2025 | GPL-3.0 (dataset card) | Somers et al., CVPRW 2024 |
| Metrica sample data | No licence; "be responsible, acknowledge the source" | Metrica Sports |
| SkillCorner open data | See repo | SkillCorner |
| StatsBomb open data | StatsBomb user agreement | StatsBomb |
| SportsMOT | CC BY-NC 4.0 | Cui et al., ICCV 2023 |
| SoccerNet-v2 camera labels | SoccerNet terms | Deliège et al., CVPRW 2021 |
| SoccerNet full broadcast videos | Non-disclosure agreement with the SoccerNet team (not redistributable) | SoccerNet |

**Models:** RF-DETR (Apache-2.0) is the default detector. Ultralytics YOLO is an optional extra (`pip install -e ".[yolo]"`) and is **AGPL-3.0**.

## Built on

[kloppy](https://github.com/PySport/kloppy) · [unravelsports](https://github.com/UnravelSports/unravelsports) · [SoccerNet](https://www.soccer-net.org/) · [Roboflow sports](https://github.com/roboflow/sports) · OpenCV

## Citation

If you use PitchLens, please cite it using [CITATION.cff](CITATION.cff) (GitHub: "Cite this repository").
Archived on Zenodo: all versions [10.5281/zenodo.24557591](https://doi.org/10.5281/zenodo.24557591); the version
used in the journal manuscript, v1.0.0, is [10.5281/zenodo.24557592](https://doi.org/10.5281/zenodo.24557592).

## Licence

Code is released under the [Apache License 2.0](LICENSE).
