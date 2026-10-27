"""EPIC 4 - Multimodal fusion & Engagement Index (0-100)."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

FUSION_COLUMNS = [
    "t", "minute", "visual", "audio", "faces", "distraction_frac",
    "label", "engagement_raw", "engagement",
]


def build_fusion(
    vision,                      # VisionResult | None
    audio_df: Optional[pd.DataFrame],
    audio_duration_s: float = 0.0,
    step_s: float = 5.0,
    w_visual: float = 0.6,
    w_audio: float = 0.4,
    ref_faces: Optional[float] = None,
    smooth_bins: int = 3,
) -> pd.DataFrame:
    """Align visual attentiveness and audio interaction on one time grid.

    visual score  = mean per-face attention x (0.6 + 0.4 x head-count coverage)
    audio score   = mean interaction weight (discussion > lecture > silence > distraction)
    engagement    = weighted blend of the available modalities, scaled to 0-100.
    """
    has_v = vision is not None and len(vision.frames) > 0
    has_a = audio_df is not None and len(audio_df) > 0
    if not has_v and not has_a:
        return pd.DataFrame(columns=FUSION_COLUMNS)

    end = max(vision.duration_s if has_v else 0.0, float(audio_duration_s) if has_a else 0.0)
    n_bins = max(1, int(np.ceil(end / step_s)))
    idx = np.arange(n_bins)
    df = pd.DataFrame({"t": idx * step_s})
    df["minute"] = df["t"] / 60.0

    # ---- visual ----
    if has_v:
        fb = (vision.frames["t"] // step_s).astype(int)
        faces = vision.frames.groupby(fb)["n_faces"].mean().reindex(idx)
        if len(vision.obs):
            ob = (vision.obs["t"] // step_s).astype(int)
            att = vision.obs.groupby(ob)["attention"].mean().reindex(idx)
        else:
            att = pd.Series(np.nan, index=idx)
        att[faces == 0] = 0.0
        ref = ref_faces if ref_faces else max(1.0, float(faces.quantile(0.9)))
        coverage = (faces / ref).clip(0, 1)
        visual = att * (0.6 + 0.4 * coverage)
        df["faces"] = faces.interpolate(limit_direction="both").values
        df["visual"] = visual.interpolate(limit_direction="both").values
    else:
        df["faces"] = np.nan
        df["visual"] = np.nan

    # ---- audio ----
    if has_a:
        ab = (audio_df["t"] // step_s).astype(int)
        grp = audio_df.assign(_b=ab, _d=(audio_df["label"] == "High Ambient Distraction").astype(float))
        a = grp.groupby("_b")["interaction"].mean().reindex(idx)
        d = grp.groupby("_b")["_d"].mean().reindex(idx)
        lab = grp.groupby("_b")["label"].agg(lambda s: s.value_counts().index[0]).reindex(idx)
        df["audio"] = a.interpolate(limit_direction="both").values
        df["distraction_frac"] = d.interpolate(limit_direction="both").values
        df["label"] = lab.ffill().bfill().values
    else:
        df["audio"] = np.nan
        df["distraction_frac"] = np.nan
        df["label"] = None

    # ---- blend ----
    v, a_ = df["visual"], df["audio"]
    wv = w_visual * v.notna().astype(float)
    wa = w_audio * a_.notna().astype(float)
    denom = (wv + wa).replace(0, np.nan)
    raw = (v.fillna(0) * wv + a_.fillna(0) * wa) / denom
    df["engagement_raw"] = (100 * raw).clip(0, 100)
    df["engagement"] = (
        df["engagement_raw"].rolling(max(1, smooth_bins), center=True, min_periods=1).mean()
    )
    return df[FUSION_COLUMNS]


def find_extrema(series: pd.Series, k: int = 3, min_sep: int = 6, highest: bool = True) -> List[int]:
    """Indices of up to ``k`` well-separated peaks (or dips) in a series."""
    vals = series.values.astype(float)
    order = np.argsort(-vals if highest else vals)
    picked: List[int] = []
    for i in order:
        if np.isnan(vals[i]):
            continue
        if all(abs(i - j) >= min_sep for j in picked):
            picked.append(int(i))
        if len(picked) >= k:
            break
    return sorted(picked)
