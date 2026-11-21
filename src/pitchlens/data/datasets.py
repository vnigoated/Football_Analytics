"""Download / load the public datasets used by PitchLens.

Licences (checked 2026-11-20, re-check before release):
- IDSSE / Sportec open data: CC-BY 4.0 (Bassek et al., Scientific Data 2025). Attribution required.
- SoccerNet SN-GSR-2025: not gated on Hugging Face; dataset card says GPL-3.0.
"""

from __future__ import annotations

from pathlib import Path

# Match IDs of the IDSSE open dataset (7 matches, 2022/23 Bundesliga + 2. Bundesliga).
IDSSE_MATCH_IDS: tuple[str, ...] = (
    "J03WMX",
    "J03WN1",
    "J03WOH",
    "J03WOY",
    "J03WPY",
    "J03WQQ",
    "J03WR9",
)

SN_GSR_REPO = "SoccerNet/SN-GSR-2025"
SN_GSR_SPLITS = ("train", "valid", "test", "challenge")


def load_idsse_tracking(
    match_id: str = "J03WPY",
    limit: int | None = None,
    only_alive: bool = True,
    sample_rate: float | None = None,
):
    """Load IDSSE tracking data (25 fps, all 22 players + ball) as a kloppy TrackingDataset."""
    if match_id not in IDSSE_MATCH_IDS:
        raise ValueError(f"Unknown IDSSE match id {match_id!r}; expected one of {IDSSE_MATCH_IDS}")
    from kloppy import sportec

    return sportec.load_open_tracking_data(
        match_id=match_id, limit=limit, only_alive=only_alive, sample_rate=sample_rate
    )


def load_idsse_events(match_id: str = "J03WPY"):
    """Load IDSSE official event data as a kloppy EventDataset."""
    if match_id not in IDSSE_MATCH_IDS:
        raise ValueError(f"Unknown IDSSE match id {match_id!r}; expected one of {IDSSE_MATCH_IDS}")
    from kloppy import sportec

    return sportec.load_open_event_data(match_id=match_id)


def download_sn_gsr(
    dest: str | Path = "data/SN-GSR-2025",
    splits: tuple[str, ...] = ("train", "valid"),
    revision: str = "main",
) -> Path:
    """Download SoccerNet Game State Reconstruction zips from Hugging Face.

    Total size is ~35 GB (train 9.8, valid 11.2, test 8.9, challenge 5.3).
    On Colab, download to /content (local SSD), not Google Drive.
    Pin ``revision`` to a commit hash for reproducible experiments.
    """
    bad = [s for s in splits if s not in SN_GSR_SPLITS]
    if bad:
        raise ValueError(f"Unknown split(s) {bad}; expected a subset of {SN_GSR_SPLITS}")
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=SN_GSR_REPO,
        repo_type="dataset",
        revision=revision,
        local_dir=str(dest),
        allow_patterns=[f"{s}.zip" for s in splits],
    )
    return Path(path)
