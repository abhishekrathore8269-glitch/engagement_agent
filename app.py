"""Multimodal Attendance & Engagement Agent - Streamlit dashboard (Skill Wallet Capstone Project 5).

100% offline edge processing. No database. No cloud calls.
Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import advisor
import audio as audio_mod
import demo
import fusion as fusion_mod
import plots
import reporting
import vision as vision_mod
from utils import AUDIO_LABELS, LABEL_COLORS, fmt_time

st.set_page_config(page_title="Multimodal Attendance & Engagement Agent",
                   page_icon="🎓", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg,#1e3a8a,#2563eb 60%,#0ea5e9); color: white;
       padding: 1.2rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;}
.hero h1 {color: white; margin: 0; font-size: 1.7rem;}
.hero p {margin: .25rem 0 0 0; opacity: .92; font-size: .95rem;}
.pill {display:inline-block; background:#dcfce7; color:#166534; padding:2px 10px; border-radius:999px;
       font-size:.75rem; font-weight:600; margin-right:6px;}
.kpi {background:white; border:1px solid #e2e8f0; border-radius:12px; padding:.8rem 1rem;}
.kpi .v {font-size:1.6rem; font-weight:700; color:#0f172a;} .kpi .l {font-size:.78rem; color:#64748b;}
.adv {border-left:5px solid #94a3b8; background:white; border-radius:8px; padding:.7rem 1rem;
      margin-bottom:.6rem; border-top:1px solid #e2e8f0; border-right:1px solid #e2e8f0; border-bottom:1px solid #e2e8f0;}
.adv.High {border-left-color:#ef4444;} .adv.Medium {border-left-color:#f59e0b;}
.adv.Info {border-left-color:#3b82f6;} .adv.Positive {border-left-color:#10b981;}
.adv .h {font-weight:700;} .adv .m {color:#334155; font-size:.93rem;} .adv .t {color:#64748b; font-size:.78rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

SEV_ICON = {"High": "🔴", "Medium": "🟠", "Info": "🔵", "Positive": "🟢"}


# --------------------------------------------------------------------------- #
# State & helpers
# --------------------------------------------------------------------------- #
def init_state() -> None:
    for k, v in {"vision": None, "audio": None, "audio_duration": 0.0,
                 "snaps": [], "snap_digest": None, "demo": False}.items():
        st.session_state.setdefault(k, v)


def clear_session() -> None:
    for k in ("vision", "audio"):
        st.session_state[k] = None
    st.session_state.update(audio_duration=0.0, snaps=[], snap_digest=None, demo=False)


def kpi(col, label, value) -> None:
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


def show_fig(fig) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def sidebar() -> SimpleNamespace:
    sb = st.sidebar
    sb.markdown("### 🔒 Privacy-first")
    sb.markdown('<span class="pill">100% offline</span><span class="pill">No face ID</span>'
                '<span class="pill">No storage</span>', unsafe_allow_html=True)
    sb.caption("Video, audio and results stay in this session's memory. Faces are tracked as "
               "anonymous IDs, never recognised. Scores describe the class, not individuals.")
    if sb.button("📦 Load demo session (synthetic)", use_container_width=True):
        st.session_state.vision = demo.make_demo_vision()
        st.session_state.audio = demo.make_demo_audio()
        st.session_state.audio_duration = float(len(st.session_state.audio))
        st.session_state.demo = True
        st.rerun()
    if sb.button("🗑️ Clear session data", use_container_width=True):
        clear_session()
        st.rerun()

    s = SimpleNamespace()
    with sb.expander("Attendance (Epic 2)", expanded=False):
        s.presence_thr = st.slider("Sustained-presence threshold (%)", 50, 95, 70, 5)
        s.sample_step = st.slider("Video sampling interval (s)", 0.5, 10.0, 2.0, 0.5)
        s.max_gap = st.slider("Track memory / re-acquire gap (s)", 2, 60, 10, 1)
        s.expected = st.number_input("Expected roster size (0 = unknown)", 0, 500, 0)
        s.min_conf = st.slider("Face detection confidence", 0.3, 0.9, 0.5, 0.05)
        s.blur = st.checkbox("Pixelate faces in preview", True)
    with sb.expander("Audio (Epic 3)", expanded=False):
        s.window = st.slider("Analysis window (s)", 0.5, 3.0, 1.0, 0.5)
        s.margin = st.slider("Silence margin above noise floor (dB)", 3, 15, 6)
        s.turn_thr = st.slider("Discussion turn sensitivity", 1.0, 5.0, 2.0, 0.5,
                               help="Lower = more windows classified as Student Discussion")
    with sb.expander("Fusion (Epic 4)", expanded=False):
        s.step = st.slider("Fusion time step (s)", 2, 30, 5)
        s.w_v = st.slider("Visual weight", 0.0, 1.0, 0.6, 0.05)
        s.w_a = round(1.0 - s.w_v, 2)
        st.caption(f"Audio weight = {s.w_a}")
        s.smooth = st.slider("Smoothing (time steps)", 1, 9, 3, 2)
    with sb.expander("Advisor (Epic 5)", expanded=False):
        s.low_thr = st.slider("Low engagement below", 20, 70, 45)
        s.low_min = st.slider("…for at least (min)", 1, 15, 5)
        s.high_thr = st.slider("Strong engagement above", 60, 95, 75)
        s.mono_min = st.slider("Max uninterrupted lecture (min)", 5, 30, 10)
        s.cooldown = st.slider("Advisory cool-down (min)", 2, 30, 10)
    return s


def compute_fusion(s):
    v, a = st.session_state.vision, st.session_state.audio
    ref = None
    if v is not None and s.expected:
        ref = float(s.expected)
    return fusion_mod.build_fusion(v, a, st.session_state.audio_duration, float(s.step),
                                   s.w_v, s.w_a, ref_faces=ref, smooth_bins=s.smooth)


def compute_attendance(s):
    v = st.session_state.vision
    if v is None:
        return None
    return vision_mod.build_attendance(v.obs, v.duration_s, v.sample_dt, s.presence_thr / 100.0,
                                       int(s.expected))


def advisor_cfg(s) -> advisor.AdvisorConfig:
    return advisor.AdvisorConfig(low_threshold=s.low_thr, low_minutes=s.low_min,
                                 high_threshold=s.high_thr, monologue_minutes=s.mono_min,
                                 cooldown_minutes=s.cooldown)


# --------------------------------------------------------------------------- #
# Tab 1 - Attendance
# --------------------------------------------------------------------------- #
def run_vision(frames_iter, total_hint, s, dt) -> bool:
    prog = st.progress(0.0, text="Starting…")
    detector = None
    ok = False
    try:
        detector = vision_mod.FaceDetector(min_conf=s.min_conf)
        res = vision_mod.analyze_samples(
            frames_iter, detector, sample_dt=dt, max_gap_s=float(s.max_gap), total_hint=total_hint,
            progress_cb=lambda f, m: prog.progress(min(1.0, f), text=m), blur_preview=s.blur)
        st.session_state.vision = res
        st.session_state.demo = False
        st.session_state.flash = f"Analysed {len(res.frames)} samples with {res.backend}."
        ok = True
    except Exception as exc:  # surfaced to the educator rather than crashing the app
        prog.empty()
        st.error(f"Vision processing failed: {exc}")
        ok = False
    finally:
        if detector is not None:
            detector.close()
    return ok


def tab_attendance(s) -> None:
    st.subheader("Automated presence-based attendance")
    st.caption("A participant is marked **Present** only if their face is tracked for more than "
               f"**{s.presence_thr}%** of the session - hallway pass-throughs never qualify.")
    src = st.radio("Video source", ["Upload classroom video", "Upload photo(s)", "Webcam snapshots"],
                   horizontal=True)

    if src == "Upload classroom video":
        up = st.file_uploader("Classroom video", type=["mp4", "avi", "mov", "mkv", "webm"])
        if up and st.button("▶ Analyse video", type="primary"):
            suffix = os.path.splitext(up.name)[1] or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                tmp.write(up.getbuffer())
                tmp.close()
                fps, dur = vision_mod.video_meta(tmp.name)
                total = int(dur / s.sample_step) + 1 if dur else None
                done = run_vision(vision_mod.iter_video_samples(tmp.name, s.sample_step), total, s, s.sample_step)
            except Exception as exc:
                done = False
                st.error(f"Could not process video: {exc}")
            finally:
                try:
                    os.unlink(tmp.name)  # raw video is never kept
                except OSError:
                    pass
            if done:
                st.rerun()

    elif src == "Upload photo(s)":
        step = st.number_input("Seconds between photos (treated as a time series)", 1.0, 600.0, 60.0)
        ups = st.file_uploader("Classroom photo(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if ups and st.button("▶ Analyse photo(s)", type="primary"):
            try:
                imgs = [vision_mod.decode_image(u.getvalue()) for u in ups]
                done = run_vision(vision_mod.image_samples(imgs, float(step)), len(imgs), s, float(step))
            except Exception as exc:
                done = False
                st.error(f"Could not process photos: {exc}")
            if done:
                st.rerun()

    else:
        step = st.number_input("Seconds between snapshots (session clock)", 1.0, 600.0, 60.0)
        shot = st.camera_input("Take a classroom snapshot")
        if shot is not None:
            data = shot.getvalue()
            dig = hashlib.md5(data).hexdigest()
            if dig != st.session_state.snap_digest:
                try:
                    st.session_state.snaps.append(vision_mod.decode_image(data))
                    st.session_state.snap_digest = dig
                except Exception as exc:
                    st.error(f"Snapshot failed: {exc}")
        st.write(f"Snapshots collected this session: **{len(st.session_state.snaps)}**")
        if st.session_state.snaps and st.button("▶ Analyse snapshots", type="primary"):
            if run_vision(vision_mod.image_samples(st.session_state.snaps, float(step)),
                          len(st.session_state.snaps), s, float(step)):
                st.rerun()

    v = st.session_state.vision
    if v is None:
        st.info("Upload a video/photos, use the webcam, or click **Load demo session** in the sidebar.")
        return
    if len(v.frames) < 5 and not st.session_state.demo:
        st.warning("Fewer than 5 samples: the sustained-presence rule needs a longer series to be meaningful.")

    att = compute_attendance(s)
    n_present = int((att["Status"] == "Present").sum()) if len(att) else 0
    c = st.columns(4)
    kpi(c[0], "Present", n_present)
    kpi(c[1], "Absent / transient", int((att["Status"] == "Absent").sum()) if len(att) else 0)
    kpi(c[2], "Session length", fmt_time(v.duration_s))
    kpi(c[3], "Detector", v.backend)

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("##### Attendance register")
        if len(att):
            st.dataframe(att, use_container_width=True, hide_index=True, height=360)
            st.download_button("⬇ Download attendance CSV", att.to_csv(index=False).encode(),
                               "attendance_register.csv", "text/csv")
        else:
            st.warning("No faces were detected in this source.")
    with right:
        if v.preview is not None:
            st.markdown("##### Preview (faces pixelated)" if s.blur else "##### Preview")
            st.image(v.preview, use_container_width=True)
        if len(att):
            show_fig(plots.plot_presence(att, s.presence_thr))


# --------------------------------------------------------------------------- #
# Tab 2 - Audio
# --------------------------------------------------------------------------- #
def tab_audio(s) -> None:
    st.subheader("Acoustic interaction & noise dynamics")
    st.caption("Spectral energy + voice-activity detection classify each window as Silence, Instructor "
               "Lecture, Student Discussion or High Ambient Distraction. Speech is never transcribed or stored.")
    up = st.file_uploader("Classroom audio (or the classroom video file, if ffmpeg is installed)",
                          type=["wav", "mp3", "ogg", "flac", "m4a", "mp4", "mov", "mkv", "webm"])
    if up and st.button("▶ Analyse audio", type="primary"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1] or ".wav")
        try:
            tmp.write(up.getbuffer())
            tmp.close()
            with st.spinner("Extracting spectral features and running VAD…"):
                y = audio_mod.load_audio(tmp.name, 16000)
                df = audio_mod.analyze_audio(y, 16000, s.window, s.margin, s.turn_thr)
            st.session_state.audio = df
            st.session_state.audio_duration = len(y) / 16000.0
            st.session_state.demo = False
            st.session_state.flash = f"Analysed {fmt_time(len(y) / 16000)} of audio in {len(df)} windows."
            done = True
        except Exception as exc:
            done = False
            st.error(f"Audio processing failed: {exc}")
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
        if done:
            st.rerun()

    df = st.session_state.audio
    if df is None:
        st.info("Upload audio or load the demo session from the sidebar.")
        return

    summ = audio_mod.summarize_audio(df)
    c = st.columns(4)
    for col, lab in zip(c, AUDIO_LABELS):
        share = float(summ.loc[summ["State"] == lab, "Share %"].sum())
        kpi(col, lab, f"{share:.0f}%")
    show_fig(plots.plot_audio_timeline(df))
    a, b = st.columns([1, 1.4])
    with a:
        show_fig(plots.plot_audio_share(df))
    with b:
        st.markdown("##### Window-level features")
        st.dataframe(df.drop(columns=["noise_floor_db"], errors="ignore"),
                     use_container_width=True, hide_index=True, height=300)
        st.download_button("⬇ Download audio windows CSV", df.to_csv(index=False).encode(),
                           "audio_windows.csv", "text/csv")


# --------------------------------------------------------------------------- #
# Tab 3 - Fusion
# --------------------------------------------------------------------------- #
def tab_fusion(s, fus) -> None:
    st.subheader("Multimodal fusion & Engagement Index")
    if fus is None or fus.empty:
        st.info("Run the video and/or audio analysis first (or load the demo session).")
        return
    st.caption("Engagement Index = weighted blend of visual attentiveness (forward head pose, upright "
               "posture, head-count coverage) and audio interaction energy, normalised to 0-100.")
    c = st.columns(4)
    kpi(c[0], "Average engagement", f"{fus['engagement'].mean():.0f}")
    kpi(c[1], "Peak", f"{fus['engagement'].max():.0f} @ {fmt_time(fus.loc[fus['engagement'].idxmax(), 't'])}")
    kpi(c[2], "Lowest", f"{fus['engagement'].min():.0f} @ {fmt_time(fus.loc[fus['engagement'].idxmin(), 't'])}")
    kpi(c[3], "Modalities", " + ".join(m for m, ok in [("Video", fus["visual"].notna().any()),
                                                       ("Audio", fus["audio"].notna().any())] if ok))
    comp = st.checkbox("Show visual / audio components", True)
    show_fig(plots.plot_engagement(fus, s.low_thr, s.high_thr, comp))
    with st.expander("Time-step fusion matrix"):
        st.dataframe(fus.round(3), use_container_width=True, hide_index=True)
        st.download_button("⬇ Download timeline CSV", fus.to_csv(index=False).encode(),
                           "engagement_timeline.csv", "text/csv")


# --------------------------------------------------------------------------- #
# Tab 4 - Advisor
# --------------------------------------------------------------------------- #
def advisory_card(r) -> str:
    return (f'<div class="adv {r["severity"]}"><div class="t">{r["time"]} · engagement {r["engagement"]:.0f}</div>'
            f'<div class="h">{SEV_ICON.get(r["severity"], "")} {r["title"]}</div>'
            f'<div class="m">{r["message"]}</div></div>')


def tab_advisor(s, fus, adv) -> None:
    st.subheader("AI teaching suggestion engine")
    if fus is None or fus.empty:
        st.info("Run the analysis first (or load the demo session).")
        return
    st.caption("Rule-based, non-intrusive prompts for the educator. They concern the whole class and "
               "never single out individual students.")
    total_min = max(float(fus["minute"].max()), 1.0)
    cur = st.slider("Session playback position (minutes) - simulates a live session",
                    0.0, total_min, total_min, 0.5)
    cur_s = cur * 60
    upto = fus[fus["t"] <= cur_s]
    now_e = float(upto["engagement"].iloc[-1]) if len(upto) else float("nan")
    c = st.columns(3)
    kpi(c[0], "Current engagement", "-" if np.isnan(now_e) else f"{now_e:.0f}")
    shown = adv[adv["time_s"] <= cur_s] if len(adv) else adv
    kpi(c[1], "Advisories so far", len(shown))
    kpi(c[2], "Time", fmt_time(cur_s))
    show_fig(plots.plot_engagement(fus, s.low_thr, s.high_thr, False, cursor_min=cur))
    if len(shown):
        st.markdown("##### Latest advisory")
        st.markdown(advisory_card(shown.iloc[-1]), unsafe_allow_html=True)
        with st.expander(f"Full advisory log ({len(shown)})", expanded=False):
            for _, r in shown.iloc[::-1].iterrows():
                st.markdown(advisory_card(r), unsafe_allow_html=True)
    else:
        st.success("No advisories yet at this point - the class looks on track.")


# --------------------------------------------------------------------------- #
# Tab 5 - Report
# --------------------------------------------------------------------------- #
def tab_report(s, fus, adv, att) -> None:
    st.subheader("Session analytics & educator report")
    if (fus is None or fus.empty) and att is None:
        st.info("Run the analysis first (or load the demo session).")
        return
    v = st.session_state.vision
    dur = max(v.duration_s if v else 0.0, st.session_state.audio_duration)
    k = reporting.compute_kpis(att, fus, st.session_state.audio, adv, dur)

    rows = [
        [("Session length", k.get("session_duration", "-")),
         ("Attendance rate", f"{k['attendance_rate']}%" if "attendance_rate" in k else "-"),
         ("Avg engagement", k.get("avg_engagement", "-")),
         ("Advisories", k.get("advisories", 0))],
        [("Verbal participation", f"{k['participation_rate']}%" if "participation_rate" in k else "-"),
         ("Visually attentive time", f"{k['visually_attentive_pct']}%" if "visually_attentive_pct" in k else "-"),
         ("Peak engagement", f"{k['peak_engagement']} @ {k['peak_time']}" if "peak_engagement" in k else "-"),
         ("Lowest engagement", f"{k['lowest_engagement']} @ {k['lowest_time']}" if "lowest_engagement" in k else "-")],
    ]
    for row in rows:
        cols = st.columns(4)
        for col, (l, val) in zip(cols, row):
            kpi(col, l, val)
        st.write("")

    if fus is not None and not fus.empty:
        a, b = st.columns([1.6, 1])
        with a:
            show_fig(plots.plot_engagement(fus, s.low_thr, s.high_thr, False))
        with b:
            show_fig(plots.plot_engagement_hist(fus))
    if st.session_state.audio is not None:
        show_fig(plots.plot_audio_timeline(st.session_state.audio))
    if adv is not None and len(adv):
        st.markdown("##### Advisories raised")
        st.dataframe(adv[["time", "severity", "title", "message"]], use_container_width=True, hide_index=True)

    md = reporting.build_markdown_report(k, att, adv)
    js = reporting.kpis_json(k, adv)
    bundle = reporting.build_zip(md, js, att, fus, st.session_state.audio, adv)
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ Report (Markdown)", md.encode(), "session_report.md", "text/markdown")
    d2.download_button("⬇ Summary (JSON)", js.encode(), "session_summary.json", "application/json")
    d3.download_button("⬇ Full bundle (ZIP)", bundle, "session_bundle.zip", "application/zip")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    init_state()
    st.markdown('<div class="hero"><h1>🎓 Multimodal Attendance & Engagement Agent</h1>'
                '<p>Offline, privacy-preserving classroom intelligence - sustained-presence attendance, '
                'acoustic dynamics, fused engagement scoring and teaching advisories.</p></div>',
                unsafe_allow_html=True)
    s = sidebar()
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    try:
        fus = compute_fusion(s)
        att = compute_attendance(s)
        adv = advisor.generate_advisories(fus, advisor_cfg(s)) if len(fus) else pd.DataFrame(columns=advisor.ADVISORY_COLUMNS)
    except Exception as exc:
        st.error(f"Could not compute session analytics: {exc}")
        fus, att, adv = None, None, pd.DataFrame(columns=advisor.ADVISORY_COLUMNS)

    t1, t2, t3, t4, t5 = st.tabs(["👥 Attendance", "🔊 Audio Dynamics", "📈 Engagement Fusion",
                                  "💡 Teaching Advisor", "📋 Session Report"])
    with t1:
        tab_attendance(s)
    with t2:
        tab_audio(s)
    with t3:
        tab_fusion(s, fus)
    with t4:
        tab_advisor(s, fus, adv)
    with t5:
        tab_report(s, fus, adv, att)


if __name__ == "__main__":
    main()
