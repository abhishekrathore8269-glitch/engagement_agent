"""EPIC 5/6 - Session KPIs and downloadable educator report."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from utils import fmt_time


def compute_kpis(attendance, fusion, audio_df, advisories, duration_s) -> dict:
    k: dict = {"session_duration": fmt_time(duration_s)}
    if attendance is not None and len(attendance):
        present = int((attendance["Status"] == "Present").sum())
        transient = int((attendance["Note"] == "Transient / pass-through").sum())
        roster = max(1, len(attendance) - transient)
        k.update(present=present, roster=roster, transient_tracks=transient,
                 attendance_rate=round(100 * present / roster, 1))
    if fusion is not None and len(fusion):
        e = fusion["engagement"]
        k.update(avg_engagement=round(float(e.mean()), 1),
                 peak_engagement=round(float(e.max()), 1),
                 peak_time=fmt_time(fusion.loc[e.idxmax(), "t"]),
                 lowest_engagement=round(float(e.min()), 1),
                 lowest_time=fmt_time(fusion.loc[e.idxmin(), "t"]),
                 pct_time_high=round(float((e >= 70).mean() * 100), 1),
                 pct_time_low=round(float((e < 45).mean() * 100), 1))
        if fusion["visual"].notna().any():
            k["visually_attentive_pct"] = round(float((fusion["visual"] >= 0.6).mean() * 100), 1)
    if audio_df is not None and len(audio_df):
        share = audio_df["label"].value_counts(normalize=True) * 100
        for lab, key in [("Student Discussion", "discussion_pct"), ("Instructor Lecture", "lecture_pct"),
                         ("Silence", "silence_pct"), ("High Ambient Distraction", "distraction_pct")]:
            k[key] = round(float(share.get(lab, 0.0)), 1)
        k["participation_rate"] = k["discussion_pct"]
    k["advisories"] = 0 if advisories is None else int(len(advisories))
    return k


def build_markdown_report(kpis, attendance, advisories, title="Classroom Session Report") -> str:
    L = [f"# {title}", f"_Generated {datetime.now():%Y-%m-%d %H:%M} - processed 100% offline; "
         "no images, audio or identities were stored._", "", "## Key indicators"]
    labels = {
        "session_duration": "Session duration", "attendance_rate": "Attendance rate (%)",
        "present": "Students present", "roster": "Roster considered",
        "transient_tracks": "Transient pass-through tracks ignored",
        "avg_engagement": "Average Engagement Index", "peak_engagement": "Peak engagement",
        "peak_time": "Peak at", "lowest_engagement": "Lowest engagement", "lowest_time": "Lowest at",
        "pct_time_high": "Time with engagement >= 70 (%)", "pct_time_low": "Time with engagement < 45 (%)",
        "visually_attentive_pct": "Time visually attentive (%)",
        "participation_rate": "Verbal participation (discussion share, %)",
        "lecture_pct": "Instructor lecture (%)", "silence_pct": "Silence (%)",
        "distraction_pct": "High ambient distraction (%)", "advisories": "Teaching advisories raised",
    }
    for key, name in labels.items():
        if key in kpis:
            L.append(f"- **{name}:** {kpis[key]}")
    L += ["", "## Teaching advisories"]
    if advisories is not None and len(advisories):
        for _, r in advisories.iterrows():
            L.append(f"- `{r['time']}` **[{r['severity']}] {r['title']}** - {r['message']}")
    else:
        L.append("- No advisories were triggered.")
    L += ["", "## Attendance register", ""]
    if attendance is not None and len(attendance):
        L.append(attendance.to_markdown(index=False) if _has_tabulate() else attendance.to_string(index=False))
    else:
        L.append("No attendance data.")
    L += ["", "---", "_Engagement scores describe the class as a whole and are intended for "
          "pedagogical reflection, not for evaluating individual students._"]
    return "\n".join(L)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except Exception:
        return False


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def kpis_json(kpis, advisories) -> str:
    payload = {"kpis": kpis,
               "advisories": [] if advisories is None else advisories.to_dict(orient="records")}
    return json.dumps(payload, indent=2, default=_json_default)


def build_zip(markdown: str, kpi_json: str, attendance, fusion, audio_df, advisories) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("session_report.md", markdown)
        z.writestr("session_summary.json", kpi_json)
        for name, df in [("attendance.csv", attendance), ("engagement_timeline.csv", fusion),
                         ("audio_windows.csv", audio_df), ("advisories.csv", advisories)]:
            if df is not None and len(df):
                z.writestr(name, df.to_csv(index=False))
    return buf.getvalue()
