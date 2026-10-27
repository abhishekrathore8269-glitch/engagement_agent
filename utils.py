"""Shared helpers and constants."""
from __future__ import annotations

from typing import List

import numpy as np

AUDIO_LABELS: List[str] = [
    "Silence",
    "Instructor Lecture",
    "Student Discussion",
    "High Ambient Distraction",
]

LABEL_COLORS = {
    "Silence": "#94a3b8",
    "Instructor Lecture": "#3b82f6",
    "Student Discussion": "#10b981",
    "High Ambient Distraction": "#ef4444",
}

# How much each acoustic state contributes to "interaction energy" (0-1) in fusion.
INTERACTION_WEIGHTS = {
    "Silence": 0.35,                 # quiet work / thinking time is neutral-low
    "Instructor Lecture": 0.65,      # one-way but on-task
    "Student Discussion": 1.00,      # active verbal interaction
    "High Ambient Distraction": 0.0, # off-task noise
}


def fmt_time(seconds) -> str:
    """Seconds -> HH:MM:SS."""
    try:
        s = int(round(max(0.0, float(seconds))))
    except (TypeError, ValueError):
        return "00:00:00"
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def safe_float(x, default=float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default
