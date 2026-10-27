"""EPIC 5 - Heuristic AI teaching-suggestion engine (rule-based, fully offline)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from utils import fmt_time

ADVISORY_COLUMNS = ["time_s", "time", "severity", "rule", "title", "message", "engagement"]


@dataclass
class AdvisorConfig:
    low_threshold: float = 45.0
    low_minutes: float = 5.0
    high_threshold: float = 75.0
    high_minutes: float = 5.0
    monologue_minutes: float = 10.0
    noise_minutes: float = 3.0
    noise_fraction: float = 0.30
    silence_minutes: float = 4.0
    cooldown_minutes: float = 10.0


def generate_advisories(fusion: pd.DataFrame, cfg: AdvisorConfig) -> pd.DataFrame:
    if fusion is None or len(fusion) == 0:
        return pd.DataFrame(columns=ADVISORY_COLUMNS)

    step = float(fusion["t"].diff().median()) if len(fusion) > 1 else 5.0
    if not np.isfinite(step) or step <= 0:
        step = 5.0

    def nb(minutes: float) -> int:
        return max(1, int(round(minutes * 60.0 / step)))

    eng = fusion["engagement"].values.astype(float)
    faces = fusion["faces"].values.astype(float)
    labels = fusion["label"].values
    dist = fusion["distraction_frac"].values.astype(float)
    has_audio = pd.notna(fusion["audio"]).any()
    has_faces = np.isfinite(faces).any()

    n3 = nb(3)
    face_roll = pd.Series(faces).rolling(n3, min_periods=n3).mean().values if has_faces else None

    last_fired: Dict[str, float] = {}
    out: List[dict] = []

    def fire(i: int, rule: str, severity: str, title: str, message: str) -> None:
        t = float(fusion["t"].iloc[i])
        if rule in last_fired and t - last_fired[rule] < cfg.cooldown_minutes * 60:
            return
        last_fired[rule] = t
        out.append({
            "time_s": t, "time": fmt_time(t), "severity": severity, "rule": rule,
            "title": title, "message": message, "engagement": round(float(eng[i]), 1),
        })

    for i in range(len(fusion)):
        # 1) sustained low engagement
        n = nb(cfg.low_minutes)
        w_low = eng[i - n + 1:i + 1] if i + 1 >= n else None
        # tolerant of brief bounces: window mean below threshold AND >=75% of steps below it
        if w_low is not None and w_low.mean() < cfg.low_threshold and (w_low < cfg.low_threshold).mean() >= 0.75:
            fire(i, "low_engagement", "High", "Sustained low engagement",
                 f"Collective engagement dropped below {cfg.low_threshold:.0f}% for "
                 f"{cfg.low_minutes:g} minutes - recommend launching a quick poll, an interactive "
                 f"discussion, or a 2-minute stretch break.")

        # 2) sharp decline
        if i + 1 >= n3 + 2:
            cur = eng[i - n3 + 1:i + 1].mean()
            prior = eng[max(0, i - n3 - nb(15) + 1): i - n3 + 1]
            if len(prior) >= n3 and prior.max() - cur >= 20 and cur < 65:
                fire(i, "sharp_decline", "Medium", "Attention is trending down",
                     f"Engagement fell about {prior.max() - cur:.0f} points in the last few minutes. "
                     "Consider a concrete example, a change of activity, or a quick comprehension check.")

        # 3) long one-way lecturing
        n = nb(cfg.monologue_minutes)
        if has_audio and i + 1 >= n:
            w = labels[i - n + 1:i + 1]
            lec = np.mean(w == "Instructor Lecture")
            disc = np.mean(w == "Student Discussion")
            if lec >= 0.85 and disc < 0.05:
                fire(i, "monologue", "Medium", "Extended one-way lecture",
                     f"About {cfg.monologue_minutes:g} minutes of uninterrupted lecturing with no student "
                     "talk. Try think-pair-share, a cold-call question, or a 60-second turn-and-talk.")

        # 4) ambient noise
        n = nb(cfg.noise_minutes)
        if has_audio and i + 1 >= n and np.nanmean(dist[i - n + 1:i + 1]) >= cfg.noise_fraction:
            fire(i, "noise", "Medium", "High ambient classroom distraction",
                 "Sustained background noise detected. Consider a brief reset: pause, "
                 "re-state the task, and set a clear volume expectation for group work.")

        # 5) prolonged silence with weak visual engagement
        n = nb(cfg.silence_minutes)
        if has_audio and i + 1 >= n:
            if np.mean(labels[i - n + 1:i + 1] == "Silence") >= 0.75 and eng[i - n + 1:i + 1].mean() < 60:
                fire(i, "silence", "Info", "Extended silence",
                     "Long quiet stretch with modest engagement. If this is not planned individual work, "
                     "invite questions or ask students to summarise the last idea to a neighbour.")

        # 6) head-count drop
        if face_roll is not None and i + 1 - n3 > 0:
            cur = face_roll[i]
            peak = np.nanmax(face_roll[: i + 1 - n3]) if np.isfinite(face_roll[: i + 1 - n3]).any() else np.nan
            if np.isfinite(cur) and np.isfinite(peak) and peak >= 3 and cur <= 0.75 * peak:
                fire(i, "headcount", "Info", "Visible head-count dropped",
                     f"Visible participants fell from about {peak:.0f} to {cur:.0f}. This may be a camera "
                     "framing issue or students leaving - a quick check-in is worthwhile.")

        # 7) positive reinforcement
        n = nb(cfg.high_minutes)
        w_hi = eng[i - n + 1:i + 1] if i + 1 >= n else None
        if w_hi is not None and w_hi.mean() >= cfg.high_threshold and (w_hi >= cfg.high_threshold).mean() >= 0.75:
            fire(i, "high_engagement", "Positive", "Strong engagement window",
                 f"Engagement stayed above {cfg.high_threshold:.0f}% for {cfg.high_minutes:g} minutes. "
                 "Good moment to introduce the most challenging concept or an extension task.")

    return pd.DataFrame(out, columns=ADVISORY_COLUMNS)
