# PitchLens: Football Analytics from Broadcast Video

[![CI](https://github.com/vnigoated/Football_Analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/vnigoated/Football_Analytics/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%E2%80%933.12-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

**PitchLens** turns ordinary TV footage of a football match into **tracking data** (where every player and the ball is, in metres). It then computes **pressing intensity** and **Expected Threat (xT)**, each reported **with honest error bars**.

> 🚧 **Status: early development (v0.1).** The analytics core works on open tracking data. The computer-vision pipeline is being built (see roadmap).

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
| `pitchlens.data` | ✅ IDSSE open tracking/events (via kloppy); SoccerNet-GSR download from Hugging Face |
| `pitchlens.analytics.pressing` | ✅ Pressing intensity (Bekkers 2025, via `unravelsports`) + per-frame summaries |
| `pitchlens.analytics.xt` | ✅ Expected Threat grid (Singh 2019), fit + rate actions |
| `pitchlens.geometry` | ✅ Pitch landmark model, RANSAC homography with reprojection error |
| `pitchlens.simulation` | ✅ Virtual broadcast camera fitted to real broadcasts (SkillCorner), player hiding, bias experiments with confidence intervals |
| `pitchlens.vision` | ⏳ Detection, tracking, teams, keypoints (planned) |

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
  analytics/    pressing intensity, xT
  simulation/   virtual broadcast camera (research core)
  vision/       detection / tracking / keypoints (WIP)
  cli.py        `pitchlens` command
tests/          unit tests (no network needed)
docs/           project overview + research plan
colab/          Colab notebooks
```

## Roadmap

| Weeks | Milestone |
|---|---|
| 1–2 | ✅ Scaffold, analytics core, IDSSE pressing baseline · detection + tracking + teams |
| 3–4 | Pitch keypoints + homography on video, replay filtering, minimap |
| 5–6 | Ball tracking, possession, pass/carry detection · visibility-bias experiment |
| 7–8 | kloppy export, xT on detected events, dashboard · imputation benchmark |
| 9–10 | Uncertainty propagation · end-to-end on SoccerNet-GSR (GS-HOTA) |
| 11–12 | Benchmarks, docs, Docker, PyPI, v1.0, paper preprint |

## Data and licences

Datasets are **never committed** to this repo; download scripts are provided instead.

| Data | Licence | Credit |
|---|---|---|
| IDSSE / Sportec open data | CC-BY 4.0 | Bassek, Rein, Weber & Memmert, *Scientific Data* (2025), doi:10.1038/s41597-025-04505-y |
| SoccerNet SN-GSR-2025 | GPL-3.0 (dataset card) | Somers et al., CVPRW 2024 |
| Metrica sample data | No licence; "be responsible, acknowledge the source" | Metrica Sports |
| SkillCorner open data | See repo | SkillCorner |
| StatsBomb open data | StatsBomb user agreement | StatsBomb |

**Models:** RF-DETR (Apache-2.0) is the default detector. Ultralytics YOLO is an optional extra (`pip install -e ".[yolo]"`) and is **AGPL-3.0**.

## Built on

[kloppy](https://github.com/PySport/kloppy) · [unravelsports](https://github.com/UnravelSports/unravelsports) · [SoccerNet](https://www.soccer-net.org/) · [Roboflow sports](https://github.com/roboflow/sports) · OpenCV

## Licence

Code is released under the [Apache License 2.0](LICENSE).
