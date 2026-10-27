"""Matplotlib figures with a clean dashboard look."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fusion import find_extrema
from utils import AUDIO_LABELS, LABEL_COLORS

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "font.size": 10, "figure.facecolor": "white",
})
BLUE, GREEN, RED, GREY = "#2563eb", "#10b981", "#ef4444", "#94a3b8"


def plot_presence(att: pd.DataFrame, threshold_pct: float):
    d = att[att["Student ID"].notna()].copy()
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.28 * len(d) + 1)))
    colors = [GREEN if s == "Present" else RED for s in d["Status"]]
    ax.barh(d["Student ID"], d["Presence %"], color=colors)
    ax.axvline(threshold_pct, color="#0f172a", ls="--", lw=1)
    ax.text(threshold_pct + 1, -0.9, f"{threshold_pct:.0f}% threshold", fontsize=8)
    ax.set_xlim(0, 100); ax.invert_yaxis()
    ax.set_xlabel("Presence (% of session)")
    fig.tight_layout()
    return fig


def plot_audio_timeline(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    t = df["t"].values / 60.0
    ax.plot(t, df["rms_db"], color="#0f172a", lw=0.7)
    lo, hi = float(df["rms_db"].min()) - 3, float(df["rms_db"].max()) + 3
    for lab in AUDIO_LABELS:
        mask = (df["label"] == lab).values
        if mask.any():
            ax.fill_between(t, lo, hi, where=mask, color=LABEL_COLORS[lab], alpha=0.35,
                            step="post", linewidth=0, label=lab)
    ax.set_ylim(lo, hi); ax.set_xlim(t.min(), t.max() + (t[1] - t[0] if len(t) > 1 else 1))
    ax.set_xlabel("Minutes"); ax.set_ylabel("Level (dB)")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    fig.tight_layout()
    return fig


def plot_audio_share(df: pd.DataFrame):
    share = df["label"].value_counts(normalize=True).reindex(AUDIO_LABELS).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(4), share.values, color=[LABEL_COLORS[l] for l in AUDIO_LABELS])
    ax.set_xticks(range(4)); ax.set_xticklabels(["Silence", "Lecture", "Discussion", "Distraction"])
    ax.set_ylabel("% of session"); ax.grid(axis="x", visible=False)
    for i, v in enumerate(share.values):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    fig.tight_layout()
    return fig


def plot_engagement(fusion: pd.DataFrame, low: float = 45.0, high: float = 75.0,
                    show_components: bool = True, cursor_min=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    x, y = fusion["minute"], fusion["engagement"]
    ax.axhspan(0, low, color=RED, alpha=0.06)
    ax.axhspan(high, 100, color=GREEN, alpha=0.06)
    if show_components:
        if fusion["visual"].notna().any():
            ax.plot(x, fusion["visual"] * 100, color=GREY, lw=1, ls=":", label="Visual attentiveness")
        if fusion["audio"].notna().any():
            ax.plot(x, fusion["audio"] * 100, color="#f59e0b", lw=1, ls=":", label="Audio interaction")
    ax.plot(x, y, color=BLUE, lw=2.4, label="Engagement Index")
    ax.fill_between(x, 0, y, color=BLUE, alpha=0.10)
    sep = max(3, len(fusion) // 8)
    for i in find_extrema(y, 3, sep, True):
        ax.scatter(x.iloc[i], y.iloc[i], color=GREEN, marker="^", s=60, zorder=5)
        ax.annotate(f"{y.iloc[i]:.0f}", (x.iloc[i], y.iloc[i]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8, color="#047857")
    for i in find_extrema(y, 3, sep, False):
        ax.scatter(x.iloc[i], y.iloc[i], color=RED, marker="v", s=60, zorder=5)
        ax.annotate(f"{y.iloc[i]:.0f}", (x.iloc[i], y.iloc[i]), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=8, color="#b91c1c")
    ax.axhline(low, color=RED, ls="--", lw=0.8)
    ax.axhline(high, color=GREEN, ls="--", lw=0.8)
    if cursor_min is not None:
        ax.axvline(cursor_min, color="#0f172a", lw=1.2)
    ax.set_ylim(0, 100); ax.set_xlim(x.min(), x.max() if x.max() > x.min() else x.min() + 1)
    ax.set_xlabel("Session time (minutes)"); ax.set_ylabel("Engagement Index (0-100)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_engagement_hist(fusion: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(fusion["engagement"], bins=np.arange(0, 105, 10), color=BLUE, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Engagement Index"); ax.set_ylabel("Time steps"); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig
