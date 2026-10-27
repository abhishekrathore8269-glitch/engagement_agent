"""EPIC 3 - Audio speech & noise processing.

Classroom audio is cut into fixed windows (default 1 s) and each window is
labelled as one of:

    Silence | Instructor Lecture | Student Discussion | High Ambient Distraction

Method (all classical DSP, runs offline, no speech is transcribed or stored):
  1. Frame-level STFT -> RMS energy (dB), spectral flatness, speech-band (200-4000 Hz) ratio.
  2. Adaptive noise floor (10th percentile of frame energy).
  3. Frame VAD = clearly above the floor AND tonal (low flatness) AND speech-band dominant.
  4. Window rules: loud non-speech / noisy -> Distraction; voiced -> speech; low energy -> Silence.
  5. Speech windows are split into Lecture vs Discussion using speaker-turn evidence
     (pitch jumps between neighbouring speech windows + pauses): frequent turns => discussion.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from utils import INTERACTION_WEIGHTS

try:  # librosa is preferred for loading/pitch, but the module degrades gracefully
    import librosa
except Exception:  # pragma: no cover
    librosa = None

N_FFT = 512
HOP = 256


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Load any audio/video file (mp3, wav, m4a, mp4 ... needs ffmpeg for compressed) as mono."""
    y = None
    if librosa is not None:
        y, _ = librosa.load(path, sr=sr, mono=True)
    else:  # minimal fallback: PCM wav only
        import wave
        with wave.open(path, "rb") as wf:
            if wf.getsampwidth() != 2:
                raise ValueError("Install librosa to read this audio format (fallback reads 16-bit WAV).")
            raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
            if wf.getnchannels() > 1:
                raw = raw.reshape(-1, wf.getnchannels()).mean(axis=1)
            native = wf.getframerate()
        if native != sr:
            idx = np.linspace(0, len(raw) - 1, int(len(raw) * sr / native))
            raw = np.interp(idx, np.arange(len(raw)), raw)
        y = raw
    if y is None or len(y) < sr:
        raise ValueError("Audio is shorter than 1 second or could not be decoded.")
    return np.asarray(y, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Frame-level features (NumPy so behaviour is identical with/without librosa)
# --------------------------------------------------------------------------- #
def _frame_features(y: np.ndarray, sr: int):
    pad = N_FFT // 2
    yp = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + (len(yp) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = yp[idx] * np.hanning(N_FFT)[None, :]
    mag = np.abs(np.fft.rfft(frames, axis=1)).T            # (freq, frames)
    power = mag ** 2
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)

    rms = np.sqrt(np.mean(frames ** 2, axis=1) * 2.0)      # compensate Hann window
    db = 20 * np.log10(rms + 1e-6)

    eps = 1e-10
    flatness = np.exp(np.mean(np.log(mag + eps), axis=0)) / (np.mean(mag, axis=0) + eps)
    centroid = (freqs[:, None] * mag).sum(axis=0) / (mag.sum(axis=0) + eps)
    band = (freqs >= 200) & (freqs <= 4000)
    band_ratio = power[band].sum(axis=0) / (power.sum(axis=0) + eps)
    return db, flatness, centroid, band_ratio


def _window_f0(seg: np.ndarray, sr: int) -> float:
    """Median fundamental frequency of a window (NaN if not estimable)."""
    try:
        if librosa is not None:
            f0 = librosa.yin(seg, fmin=70, fmax=400, sr=sr, frame_length=1024)
            return float(np.median(f0))
        # autocorrelation fallback on the loudest 64 ms chunk
        n = int(0.064 * sr)
        start = int(np.argmax(np.convolve(seg ** 2, np.ones(n), mode="valid")))
        x = seg[start:start + n] - np.mean(seg[start:start + n])
        ac = np.correlate(x, x, mode="full")[len(x) - 1:]
        lo, hi = int(sr / 400), int(sr / 70)
        lag = lo + int(np.argmax(ac[lo:hi]))
        return float(sr / lag) if ac[lag] > 0.3 * ac[0] else float("nan")
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #
def analyze_audio(
    y: np.ndarray,
    sr: int = 16000,
    window_s: float = 1.0,
    silence_margin_db: float = 6.0,
    turn_threshold: float = 2.0,
) -> pd.DataFrame:
    """Return one row per window with features, a ``label`` and an ``interaction`` score (0-1)."""
    y = np.asarray(y, dtype=np.float32)
    wlen = int(window_s * sr)
    n_win = len(y) // wlen
    if n_win < 1:
        raise ValueError("Audio is shorter than one analysis window.")

    db, flat, cent, band_ratio = _frame_features(y, sr)
    floor = float(np.percentile(db, 10))
    active = db > floor + silence_margin_db
    voiced = active & (flat < 0.25) & (band_ratio > 0.5)

    fpw = max(1, int(round(wlen / HOP)))
    need = n_win * fpw
    if len(db) < need:  # pad (rare edge case)
        padn = need - len(db)
        db, flat, cent, band_ratio = (np.pad(a, (0, padn), mode="edge") for a in (db, flat, cent, band_ratio))
        active = np.pad(active, (0, padn))
        voiced = np.pad(voiced, (0, padn))

    def W(a):
        return a[:need].reshape(n_win, fpw)

    mean_db = W(db).mean(axis=1)
    db_std = W(db).std(axis=1)
    flat_w = W(flat).mean(axis=1)
    cent_w = W(cent).mean(axis=1)
    band_w = W(band_ratio).mean(axis=1)
    act_frac = W(active.astype(float)).mean(axis=1)
    voi_frac = W(voiced.astype(float)).mean(axis=1)

    label = np.full(n_win, "Silence", dtype=object)
    is_speech = np.zeros(n_win, dtype=bool)
    loud = mean_db > floor + 25
    for i in range(n_win):
        if act_frac[i] < 0.15:
            label[i] = "Silence"
        elif loud[i] and (voi_frac[i] < 0.30 or flat_w[i] > 0.30):
            label[i] = "High Ambient Distraction"
        elif voi_frac[i] >= 0.25:
            is_speech[i] = True
            label[i] = "Instructor Lecture"      # refined below
        elif mean_db[i] > floor + 12:
            label[i] = "High Ambient Distraction"
        else:
            label[i] = "Silence"

    # ---- Lecture vs discussion via speaker-turn evidence ----
    f0 = np.full(n_win, np.nan)
    for i in np.where(is_speech)[0]:
        f0[i] = _window_f0(y[i * wlen:(i + 1) * wlen], sr)

    turns = np.zeros(n_win)
    prev: Optional[int] = None
    for i in np.where(is_speech)[0]:
        if prev is not None:
            gap = i - prev
            if gap <= 2 and np.isfinite(f0[i]) and np.isfinite(f0[prev]) and f0[i] > 0 and f0[prev] > 0:
                semitones = 12 * abs(np.log2(f0[i] / f0[prev]))
                if semitones > 3.5:
                    turns[i] += 1.0           # pitch jump => probable speaker change
            elif gap >= 3:
                turns[i] += 0.5               # new burst after a pause (Q&A rhythm)
        prev = i
    ctx = np.convolve(turns, np.ones(17), mode="same")
    for i in np.where(is_speech)[0]:
        if ctx[i] >= turn_threshold:
            label[i] = "Student Discussion"

    return pd.DataFrame({
        "t": np.arange(n_win) * window_s,
        "rms_db": mean_db.round(2),
        "noise_floor_db": round(floor, 2),
        "flatness": flat_w.round(4),
        "centroid_hz": cent_w.round(1),
        "speech_band_ratio": band_w.round(3),
        "active_frac": act_frac.round(3),
        "voiced_frac": voi_frac.round(3),
        "db_std": db_std.round(2),
        "f0_hz": np.round(f0, 1),
        "label": label,
        "interaction": [INTERACTION_WEIGHTS[l] for l in label],
    })


def summarize_audio(df: pd.DataFrame) -> pd.DataFrame:
    """Time share per acoustic state."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["State", "Windows", "Share %"])
    vc = df["label"].value_counts()
    out = pd.DataFrame({"State": vc.index, "Windows": vc.values})
    out["Share %"] = (100 * out["Windows"] / out["Windows"].sum()).round(1)
    return out
