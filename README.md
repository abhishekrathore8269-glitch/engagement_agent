# 🎓 Multimodal Attendance & Engagement Agent
Skill Wallet Capstone – Project 5. Offline, privacy-preserving classroom analytics in Streamlit.

## Run
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.9 – 3.12
pip install -r requirements.txt   # works on Python 3.9 - 3.14
streamlit run app.py
```
Click **Load demo session** in the sidebar to explore every tab with no camera, mic or files.
Tests: `python tests/test_core.py` (or `pytest -q`).

## Deploy on Streamlit Community Cloud
Push this folder to GitHub → New app → main file `app.py`. In **Advanced settings** pick Python 3.11 (recommended: enables MediaPipe). On Python 3.13+ MediaPipe is skipped and the app uses OpenCV Haar detection automatically. `packages.txt` installs ffmpeg/libGL.
No database, secrets or external services are needed.

## Architecture
| Module | Epic | Role |
|---|---|---|
| `vision.py` | 2 | MediaPipe face detection (OpenCV Haar fallback), centroid tracker, head-pose/posture attention, sustained-presence attendance |
| `audio.py` | 3 | STFT features, adaptive noise floor, VAD, Silence / Lecture / Discussion / Distraction classifier |
| `fusion.py` | 4 | Time-grid alignment, Engagement Index 0-100, peak/dip detection |
| `advisor.py` | 5 | Rule-based teaching advisories with cool-downs |
| `reporting.py` | 5/6 | KPIs, Markdown/JSON report, ZIP bundle |
| `plots.py`, `app.py`, `demo.py`, `tests/` | 6 | Matplotlib figures, 5-tab dashboard, synthetic session, unit tests |

## Method notes & honest limitations
* **Attendance**: Present only if tracked for **> 70 %** of sampled session time (configurable). Tracks have a re-acquire gap (default 10 s) because seated students don't move. Tracking is by position, so a student who changes seats may be split into two IDs. IDs are anonymous (`S01…`) – there is no face recognition; to report named absentees, enter the expected roster size.
* **Head pose** is approximated from MediaPipe's 6 face keypoints (eye/nose/mouth geometry), not a full 3-D solve. The Haar fallback has no keypoints and uses a neutral prior.
* **Audio** uses classical DSP heuristics. Lecture vs Discussion is inferred from pitch jumps and pauses (speaker-turn evidence), not speaker identification – tune *Discussion turn sensitivity* per room. Calibrate on a sample of your own classroom.
* **Fusion** assumes video and audio start together. Weights are adjustable in the sidebar.
* Engagement scores are collective and heuristic – use them for reflection, not to judge individual students.
* Live webcam is supported as periodic snapshots (`st.camera_input`) to keep deployment dependency-free.
