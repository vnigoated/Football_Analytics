"""Computer-vision pipeline (detection, tracking, teams, pitch keypoints).

Planned modules (see README roadmap, weeks 1-6):
    detector.py   RF-DETR (Apache-2.0) default; Ultralytics YOLO optional (AGPL-3.0)
    ball.py       small-object ball detector + Kalman/interpolation
    tracker.py    ByteTrack / BoT-SORT with re-ID
    teams.py      SigLIP embeddings -> UMAP -> KMeans team assignment
    keypoints.py  32-keypoint pitch model -> geometry.homography
    shots.py      replay / close-up detection
Install with:  pip install -e ".[vision]"
"""
