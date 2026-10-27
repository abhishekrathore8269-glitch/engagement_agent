"""EPIC 2 - Face tracking & sustained-presence attendance.

Privacy notes
-------------
* Everything runs locally. No frame ever leaves the process.
* Faces are *detected and tracked* (anonymous IDs such as ``S01``); there is no
  face recognition, embedding storage or identity database.
* Only one annotated preview frame is retained, with faces pixelated by default.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import fmt_time

try:  # pragma: no cover - import guard
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:  # pragma: no cover - import guard
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None

OBS_COLUMNS = ["t", "track", "cx", "cy", "w", "h", "yaw", "pitch", "has_pose"]


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    x: float
    y: float
    w: float
    h: float
    score: float = 1.0
    keypoints: Optional[np.ndarray] = None  # (6, 2) pixel coords (MediaPipe only)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


@dataclass
class VisionResult:
    obs: pd.DataFrame          # one row per (sample time, tracked face) incl. attention
    frames: pd.DataFrame       # one row per sampled frame: t, n_faces
    duration_s: float
    sample_dt: float
    backend: str
    preview: Optional[np.ndarray] = None  # RGB image with pixelated faces


# --------------------------------------------------------------------------- #
# Face detection
# --------------------------------------------------------------------------- #
class FaceDetector:
    """MediaPipe face detector with an OpenCV Haar-cascade fallback."""

    def __init__(self, min_conf: float = 0.5, prefer_mediapipe: bool = True):
        if cv2 is None:
            raise RuntimeError("OpenCV is not installed (pip install opencv-python-headless).")
        self.backend = "none"
        self._mp = None
        self._haar = None
        if prefer_mediapipe and mp is not None:
            try:
                # model_selection=1 -> full-range model, better for rooms / small faces
                self._mp = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=min_conf
                )
                self.backend = "mediapipe"
            except Exception:
                self._mp = None
        if self._mp is None:
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if cascade.empty():
                raise RuntimeError("No face detector backend available.")
            self._haar = cascade
            self.backend = "opencv-haar"

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        h, w = bgr.shape[:2]
        out: List[Detection] = []
        if self._mp is not None:
            res = self._mp.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            for d in res.detections or []:
                rb = d.location_data.relative_bounding_box
                x, y = max(0.0, rb.xmin * w), max(0.0, rb.ymin * h)
                bw, bh = min(rb.width * w, w - x), min(rb.height * h, h - y)
                if bw < 8 or bh < 8:
                    continue
                kps = np.array(
                    [[k.x * w, k.y * h] for k in d.location_data.relative_keypoints],
                    dtype=float,
                )
                score = float(d.score[0]) if len(d.score) else 1.0
                out.append(Detection(x, y, bw, bh, score, kps if len(kps) >= 4 else None))
        else:
            gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
            for (x, y, bw, bh) in self._haar.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
            ):
                out.append(Detection(float(x), float(y), float(bw), float(bh)))
        return out

    def close(self) -> None:
        if self._mp is not None:
            try:
                self._mp.close()
            except Exception:
                pass


def pose_ratios(kps: Optional[np.ndarray]) -> Tuple[float, float, bool]:
    """Approximate head pose from the 6 MediaPipe face-detection keypoints.

    yaw   : horizontal nose offset from the eye midpoint, in eye-distance units
            (0 = facing the camera, |yaw| grows as the head turns sideways).
    pitch : nose position between the eye line (0) and the mouth (1);
            shifts when the head tilts up/down.
    """
    if kps is None or len(kps) < 4:
        return 0.0, 0.5, False
    eyes = kps[:2]
    nose, mouth = kps[2], kps[3]
    left, right = sorted(eyes, key=lambda p: p[0])
    eye_mid = (left + right) / 2.0
    eye_d = max(right[0] - left[0], 1e-3)
    yaw = float((nose[0] - eye_mid[0]) / eye_d)
    denom = mouth[1] - eye_mid[1]
    pitch = float((nose[1] - eye_mid[1]) / denom) if abs(denom) > 1e-3 else 0.5
    return yaw, float(np.clip(pitch, -1.0, 2.0)), True


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #
class CentroidTracker:
    """Greedy centroid tracker tuned for *seated* classrooms.

    Students rarely move, so a track may "sleep" for ``max_gap_s`` (missed
    detections, head turned away) and still be re-acquired at the same spot.
    A hallway pass-through creates a short-lived track that never reaches the
    sustained-presence threshold.
    """

    def __init__(self, max_gap_s: float = 10.0, match_factor: float = 0.8):
        self.max_gap_s = max_gap_s
        self.match_factor = match_factor
        self.tracks: Dict[int, dict] = {}
        self._next = 1

    def update(self, t: float, dets: List[Detection]) -> List[int]:
        ids: List[Optional[int]] = [None] * len(dets)
        pairs = []
        for tid, tr in self.tracks.items():
            if t - tr["last_t"] > self.max_gap_s:
                continue
            for di, d in enumerate(dets):
                dist = math.hypot(d.cx - tr["cx"], d.cy - tr["cy"]) / max(tr["w"], d.w, 1.0)
                if dist <= self.match_factor:
                    pairs.append((dist, tid, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            d, tr = dets[di], self.tracks[tid]
            tr["cx"] = 0.6 * tr["cx"] + 0.4 * d.cx
            tr["cy"] = 0.6 * tr["cy"] + 0.4 * d.cy
            tr["w"] = 0.6 * tr["w"] + 0.4 * d.w
            tr["last_t"] = t
            ids[di] = tid
        for di, d in enumerate(dets):
            if ids[di] is None:
                tid = self._next
                self._next += 1
                self.tracks[tid] = {"cx": d.cx, "cy": d.cy, "w": d.w, "last_t": t}
                ids[di] = tid
        return [int(i) for i in ids]  # type: ignore[arg-type]


def track_label(tid: int) -> str:
    return f"S{tid:02d}"


# --------------------------------------------------------------------------- #
# Frame sources
# --------------------------------------------------------------------------- #
def _resize_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def video_meta(path: str) -> Tuple[float, float]:
    """Return (fps, duration_s)."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError("Could not open the video file (unsupported codec?).")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        return float(fps), float(n / fps) if n > 0 else 0.0
    finally:
        cap.release()


def iter_video_samples(
    path: str, step_s: float = 2.0, max_width: int = 960, max_samples: int = 1500
) -> Iterator[Tuple[float, np.ndarray]]:
    """Yield (timestamp_s, BGR frame) every ``step_s`` seconds without decoding every frame."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file (unsupported codec?).")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stride = max(1, int(round(fps * step_s)))
        idx, yielded = 0, 0
        while yielded < max_samples:
            if not cap.grab():
                break
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    yield idx / fps, _resize_max_width(frame, max_width)
                    yielded += 1
            idx += 1
    finally:
        cap.release()


def decode_image(data: bytes, max_width: int = 1280) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the image.")
    return _resize_max_width(img, max_width)


def image_samples(images: List[np.ndarray], step_s: float) -> Iterator[Tuple[float, np.ndarray]]:
    for i, img in enumerate(images):
        yield i * step_s, img


# --------------------------------------------------------------------------- #
# Annotation (privacy-preserving preview)
# --------------------------------------------------------------------------- #
def annotate_frame(
    frame: np.ndarray, dets: List[Detection], ids: List[int], blur: bool = True
) -> np.ndarray:
    img = frame.copy()
    H, W = img.shape[:2]
    for d, tid in zip(dets, ids):
        x0, y0 = int(max(0, d.x)), int(max(0, d.y))
        x1, y1 = int(min(W, d.x + d.w)), int(min(H, d.y + d.h))
        if x1 <= x0 or y1 <= y0:
            continue
        if blur:
            roi = img[y0:y1, x0:x1]
            small = cv2.resize(roi, (max(1, (x1 - x0) // 10), max(1, (y1 - y0) // 10)))
            img[y0:y1, x0:x1] = cv2.resize(
                small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST
            )
        cv2.rectangle(img, (x0, y0), (x1, y1), (37, 99, 235), 2)
        cv2.putText(img, track_label(tid), (x0, max(14, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (37, 99, 235), 2, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# Attentiveness scoring
# --------------------------------------------------------------------------- #
def score_attention(obs: pd.DataFrame) -> pd.DataFrame:
    """Add ``attention`` (0-1) from forward head pose, pitch and upright posture.

    * yaw score   : facing the front of the room (not turned sideways)
    * pitch score : deviation from the class's own median (looking down/up)
    * posture     : face centre dropping below the student's own baseline (slumping)
    """
    o = obs.copy()
    if o.empty:
        o["attention"] = pd.Series(dtype=float)
        return o
    pose = o["has_pose"].astype(bool)
    pitch_med = float(o.loc[pose, "pitch"].median()) if pose.any() else 0.5
    yaw_s = np.exp(-((o["yaw"].astype(float) / 0.35) ** 2))
    pitch_s = np.exp(-(((o["pitch"].astype(float) - pitch_med) / 0.18) ** 2))
    base_cy = o.groupby("track")["cy"].transform("median")
    slump = ((o["cy"] - base_cy) / o["h"].clip(lower=1.0)).clip(lower=0.0)
    posture_s = np.exp(-((slump / 0.5) ** 2))
    yaw_s = yaw_s.where(pose, 0.75)       # no keypoints (Haar): neutral prior
    pitch_s = pitch_s.where(pose, 0.75)
    # Facing the front gates the score; pitch/posture modulate it (looking down or slumping
    # lowers attention but a head turned fully away is ~0 regardless of posture).
    o["attention"] = (yaw_s * (0.4 + 0.6 * (0.6 * pitch_s + 0.4 * posture_s))).clip(0, 1)
    return o


# --------------------------------------------------------------------------- #
# Main analysis loop
# --------------------------------------------------------------------------- #
def analyze_samples(
    samples: Iterable[Tuple[float, np.ndarray]],
    detector: FaceDetector,
    sample_dt: float,
    max_gap_s: float = 10.0,
    total_hint: Optional[int] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    blur_preview: bool = True,
) -> VisionResult:
    tracker = CentroidTracker(max_gap_s=max_gap_s)
    rows: List[dict] = []
    frame_rows: List[dict] = []
    preview, best_n = None, -1

    for k, (t, frame) in enumerate(samples):
        dets = detector.detect(frame)
        ids = tracker.update(t, dets)
        frame_rows.append({"t": float(t), "n_faces": len(dets)})
        for d, tid in zip(dets, ids):
            yaw, pitch, has = pose_ratios(d.keypoints)
            rows.append({
                "t": float(t), "track": track_label(tid), "cx": d.cx, "cy": d.cy,
                "w": d.w, "h": d.h, "yaw": yaw, "pitch": pitch, "has_pose": has,
            })
        if len(dets) > best_n:
            best_n = len(dets)
            preview = annotate_frame(frame, dets, ids, blur_preview)
        if progress_cb:
            frac = min(1.0, (k + 1) / total_hint) if total_hint else 0.0
            progress_cb(frac, f"Analysed sample {k + 1} (t={fmt_time(t)}) - {len(dets)} faces")

    if not frame_rows:
        raise ValueError("No frames could be read from the selected source.")

    frames = pd.DataFrame(frame_rows)
    obs = score_attention(pd.DataFrame(rows, columns=OBS_COLUMNS))
    duration = float(frames["t"].max() + sample_dt)
    return VisionResult(obs, frames, duration, float(sample_dt), detector.backend, preview)


# --------------------------------------------------------------------------- #
# Sustained-presence attendance
# --------------------------------------------------------------------------- #
ATTENDANCE_COLUMNS = [
    "Student ID", "First Seen", "Last Seen", "Total Duration",
    "Presence %", "Status", "Note",
]


def build_attendance(
    obs: pd.DataFrame,
    session_duration_s: float,
    sample_dt: float,
    threshold: float = 0.70,
    expected_roster: int = 0,
    transient_cutoff: float = 0.25,
) -> pd.DataFrame:
    """Present only if detected for > ``threshold`` of the session.

    Duration = number of distinct sampled instants the track was seen x sampling interval,
    so brief hallway pass-throughs (a few samples) can never reach the threshold.
    """
    rows = []
    duration = max(float(session_duration_s), 1e-6)
    if not obs.empty:
        for tid, g in obs.groupby("track"):
            detected = min(g["t"].nunique() * sample_dt, duration)
            ratio = detected / duration
            status = "Present" if ratio > threshold else "Absent"
            note = ""
            if ratio < transient_cutoff:
                note = "Transient / pass-through"
            elif status == "Absent":
                note = "Below sustained-presence threshold"
            rows.append({
                "Student ID": tid,
                "First Seen": fmt_time(g["t"].min()),
                "Last Seen": fmt_time(g["t"].max()),
                "Total Duration": fmt_time(detected),
                "Presence %": round(100 * ratio, 1),
                "Status": status,
                "Note": note,
            })
    df = pd.DataFrame(rows, columns=ATTENDANCE_COLUMNS)
    if not df.empty:
        df = df.sort_values("Student ID").reset_index(drop=True)

    # Roster placeholders: anonymous system -> unseen slots are reported as Absent.
    regular = int((df["Note"] != "Transient / pass-through").sum()) if not df.empty else 0
    if expected_roster and expected_roster > regular:
        extra = [{
            "Student ID": f"UNSEEN-{i + 1:02d}", "First Seen": "-", "Last Seen": "-",
            "Total Duration": fmt_time(0), "Presence %": 0.0, "Status": "Absent",
            "Note": "Expected but never detected",
        } for i in range(expected_roster - regular)]
        df = pd.concat([df, pd.DataFrame(extra, columns=ATTENDANCE_COLUMNS)], ignore_index=True)
    return df
