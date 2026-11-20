# PitchLens: Turning Football TV Footage into Match Data

> Working name. Change it whenever you like.

## The idea in one sentence

You give it a normal TV recording of a football match, and it tells you **where every player and the ball was at every moment**. Then it uses that to measure **how dangerous each pass was** and **how hard each team was pressing**.

---

## Why this matters

Professional clubs pay companies (such as SkillCorner, Second Spectrum and Hawk-Eye) a lot of money for *tracking data*: the x/y position of every player, many times per second. That data drives scouting, tactics and fitness work.

Most teams, academies, women's leagues and lower divisions can't afford it. They do have TV or streaming footage. **This project tries to get useful tracking data out of that footage for free, and tells you honestly how accurate it is.**

---

## How it works (step by step)

Think of it as an assembly line. Each step hands its result to the next one.

```
 TV video
    │
    ▼
 1. Spot the people and the ball      →  "there is a player here, the ball is there"
    │
    ▼
 2. Follow them over time             →  "this is the same player as in the last frame"
    │
    ▼
 3. Split them into teams             →  "red shirts = Team A, blue = Team B, yellow = referee"
    │
    ▼
 4. Understand where the camera looks →  "this part of the screen is the left penalty box"
    │
    ▼
 5. Convert screen → real pitch       →  "player #7 is 12 m from goal, 5 m from the left touchline"
    │
    ▼
 6. Clean the data                    →  remove jumps and noise, skip replays and close-ups
    │
    ▼
 7. Work out what happened            →  "a pass from #7 to #10", "a dribble forward"
    │
    ▼
 8. Calculate football metrics        →  danger of each pass (xT), pressing intensity
    │
    ▼
 Outputs: data files + top-down mini-map video + dashboard
```

### What each step means in plain words

| Step | What it does | Simple analogy |
|---|---|---|
| **1. Detection** | An AI model draws boxes around players, referees and the ball in each frame. | Like face detection on your phone camera, but for footballers. |
| **2. Tracking** | Links the boxes across frames so each player keeps the same ID. | Following one friend through a crowd. |
| **3. Team assignment** | Groups players by shirt colour and look. | Sorting socks by colour. |
| **4. Pitch keypoints** | Finds known spots on the pitch (corners, penalty spots, centre circle). | Using landmarks to work out where you are on a map. |
| **5. Homography** | Uses those spots to "flatten" the TV view into a top-down pitch. | Like straightening a photo of a document you took at an angle. |
| **6. Cleaning** | Smooths positions, removes impossible jumps, ignores replays. | Editing out the shaky bits of a video. |
| **7. Events** | Works out passes and dribbles from who has the ball. | Commentary, but done by a computer. |
| **8. Metrics** | Turns positions and events into numbers coaches care about. | The stats shown on TV after the match. |

---

## The two football metrics, explained simply

### Expected Threat (xT): "How dangerous was that move?"

Split the pitch into a grid. Each square gets a number: **the chance that having the ball there leads to a goal soon**. Squares near the opponent's goal score high; your own half scores low.

- A pass that moves the ball from a low square to a high square **adds threat**.
- This credits players who move the ball into dangerous areas, not only the ones who score.

### Pressing intensity: "How much pressure is the defending team applying?"

For each attacker, estimate **how quickly the nearest defenders could reach them**, using how close they are, how fast they're moving and in which direction.

- Defenders close and running at you → **high pressure**.
- Defenders far away or moving away → **low pressure**.
- Added up over time, this shows which teams press high, when they press, and who works hardest.

---

## What you get at the end

1. **Tracking data file**: positions of every visible player and the ball over time, in metres, in a standard format that other football analytics tools can read.
2. **Mini-map video**: a top-down animated pitch next to the original footage.
3. **Metrics tables**: xT for every pass and dribble, pressing intensity per team, per player and per minute.
4. **Dashboard**: charts and heatmaps you can explore in a browser.
5. **Accuracy report**: how close our numbers are to the real (ground-truth) data. **This is what makes it trustworthy.**

---

## What makes this "industry-level" and not just a demo

- **It measures its own accuracy** on a public benchmark (SoccerNet Game State Reconstruction) instead of just looking good in a video.
- **It handles real-world mess**: replays, close-ups, camera cuts and players leaving the screen.
- **It uses standard formats**, so analysts can use the output in tools they already have.
- **It's built like real software**: installable package, command-line tool, tests, Docker, documentation.
- **It uses licences that allow commercial use** (Apache 2.0 models by default).

---

## The big honest problem (and the research opportunity)

A TV camera **only shows about 14 of the 22 players at any moment**. The rest are off screen.

This matters: if you compute pressing with some defenders missing, the numbers come out wrong. That leads straight to the research questions:

- **How wrong** do the metrics get when players are missing?
- Can we **estimate where the missing players probably are**?
- Can we put **error bars** on every number, so a coach knows how much to trust it?

The research side of the project tackles exactly these questions.

---

## Tools and data we'll use

| Purpose | Tool / dataset |
|---|---|
| Player and ball detection | RF-DETR (Apache 2.0); YOLO as an option |
| Tracking | ByteTrack / BoT-SORT |
| Team assignment | SigLIP + clustering |
| Pitch calibration | Keypoint model + homography (optionally PnLCalib) |
| Data format | kloppy (standard football data library) |
| Pressing intensity | unravelsports |
| xT | Grid model trained on StatsBomb open event data |
| Benchmark | SoccerNet Game State Reconstruction |
| Missing-player study | SkillCorner open tracking data |
| Training | Google Colab (A100 GPU) |

---

## Roadmap (about 12 weeks)

| Weeks | Goal |
|---|---|
| 1–2 | Project setup, data access, detect and track players, split teams |
| 3–4 | Pitch calibration, top-down mini-map, handling replays |
| 5–6 | Ball tracking, who has the ball, detecting passes |
| 7–8 | Export standard data, compute xT and pressing, build dashboard |
| 9–10 | Research: missing-player bias and filling in off-screen players |
| 11–12 | Benchmarks, documentation, release v1.0, write-up |

---

## Important rules

- **Never upload TV match footage to the repo.** It's copyrighted.
- **SoccerNet data:** the datasets we use (GSR, Calibration, Tracking) are openly downloadable from Hugging Face, but some others (raw full-match videos) need an NDA. Follow each dataset card's licence, and use download scripts instead of re-uploading the data.
- Publish code, training recipes and results, not other people's data.
