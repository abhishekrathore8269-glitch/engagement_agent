"""Synthetic 40-minute session so the whole app can be explored with no camera/mic/files."""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import INTERACTION_WEIGHTS
from vision import OBS_COLUMNS, VisionResult, score_attention, track_label

# (minute, engagement-ish level 0-1) keypoints driving attentiveness in the video
_PROFILE_T = [0, 5, 10, 14, 18, 22, 24, 30, 33, 36, 40]
_PROFILE_V = [0.78, 0.82, 0.72, 0.40, 0.32, 0.62, 0.86, 0.80, 0.50, 0.36, 0.55]

# (start_min, end_min, probabilities for [Silence, Lecture, Discussion, Distraction])
_AUDIO_SCRIPT = [
    (0, 12, [0.10, 0.85, 0.05, 0.00]),
    (12, 20, [0.25, 0.35, 0.00, 0.40]),
    (20, 27, [0.10, 0.20, 0.70, 0.00]),
    (27, 33, [0.10, 0.80, 0.10, 0.00]),
    (33, 38, [0.40, 0.25, 0.00, 0.35]),
    (38, 40, [0.10, 0.20, 0.70, 0.00]),
]
_STATES = ["Silence", "Instructor Lecture", "Student Discussion", "High Ambient Distraction"]


def make_demo_vision(seed: int = 7, minutes: int = 40, dt: float = 10.0) -> VisionResult:
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 / dt)
    positions = [(120 + 150 * c, 160 + 120 * r) for r in range(4) for c in range(6)]
    # (arrival_s, departure_s) - 18 regulars, 2 late, 1 early leaver, 1 tardy-late, rest absent
    spans = [(0, minutes * 60)] * 18
    spans += [(9 * 60, minutes * 60), (16 * 60, minutes * 60), (0, 20 * 60), (0, minutes * 60)]
    rows = []
    for k in range(n):
        t = k * dt
        level = float(np.interp(t / 60, _PROFILE_T, _PROFILE_V))
        for s, (arr, dep) in enumerate(spans):
            if not (arr <= t < dep) or rng.random() > 0.96:
                continue
            spread = 0.10 + (1 - level) * 0.9
            cx, cy = positions[s]
            slump = max(0.0, (0.55 - level)) * rng.random() * 70
            rows.append({
                "t": t, "track": track_label(s + 1),
                "cx": cx + rng.normal(0, 3), "cy": cy + rng.normal(0, 3) + slump,
                "w": 48.0, "h": 48.0,
                "yaw": rng.normal(0, spread * 0.55),
                "pitch": 0.55 + rng.normal(0, spread * 0.22),
                "has_pose": True,
            })
    # two hallway pass-throughs (3 samples each)
    for j, start in enumerate([6 * 60, 27 * 60]):
        for q in range(3):
            rows.append({"t": start + q * dt, "track": track_label(40 + j),
                         "cx": 900.0, "cy": 80.0 + 40 * q, "w": 40.0, "h": 40.0,
                         "yaw": 0.1, "pitch": 0.55, "has_pose": True})
    obs = score_attention(pd.DataFrame(rows, columns=OBS_COLUMNS))
    frames = obs.groupby("t").size().reindex(np.arange(n) * dt, fill_value=0).rename_axis("t")
    frames = frames.rename("n_faces").reset_index()
    return VisionResult(obs, frames, n * dt, dt, "demo (synthetic data)", None)


def make_demo_audio(seed: int = 11, minutes: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    total = minutes * 60
    labels = []
    t = 0
    while t < total:
        probs = next((p for a, b, p in _AUDIO_SCRIPT if a * 60 <= t < b * 60), _AUDIO_SCRIPT[-1][2])
        state = rng.choice(_STATES, p=probs)
        blen = int(rng.integers(5, 16))
        labels += [state] * blen
        t += blen
    labels = labels[:total]
    base_db = {"Silence": -56, "Instructor Lecture": -24, "Student Discussion": -27,
               "High Ambient Distraction": -14}
    flat = {"Silence": 0.35, "Instructor Lecture": 0.06, "Student Discussion": 0.09,
            "High Ambient Distraction": 0.45}
    df = pd.DataFrame({
        "t": np.arange(total, dtype=float),
        "rms_db": [base_db[l] + rng.normal(0, 2) for l in labels],
        "noise_floor_db": -58.0,
        "flatness": [max(0.01, flat[l] + rng.normal(0, 0.02)) for l in labels],
        "centroid_hz": [1600 + rng.normal(0, 150) if l != "High Ambient Distraction" else 2800 + rng.normal(0, 300) for l in labels],
        "speech_band_ratio": [0.75 if l in _STATES[1:3] else 0.4 for l in labels],
        "active_frac": [0.05 if l == "Silence" else 0.9 for l in labels],
        "voiced_frac": [0.7 if l in _STATES[1:3] else 0.1 for l in labels],
        "db_std": 4.0,
        "f0_hz": np.nan,
        "label": labels,
    })
    df["interaction"] = df["label"].map(INTERACTION_WEIGHTS)
    return df
