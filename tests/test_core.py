"""Epic 6 - automated tests. Run:  python -m pytest -q   (or: python tests/test_core.py)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import advisor, audio, demo, fusion, reporting, vision


def _det(x, y, w=40):
    return vision.Detection(x, y, w, w)


def test_tracker_keeps_identity_and_reacquires():
    tr = vision.CentroidTracker(max_gap_s=10)
    a = tr.update(0, [_det(100, 100), _det(300, 100)])
    b = tr.update(2, [_det(102, 101), _det(301, 99)])
    assert a == b
    c = tr.update(8, [_det(101, 100)])          # second student missed for 6 s
    d = tr.update(10, [_det(301, 100)])
    assert c[0] == a[0] and d[0] == a[1]
    e = tr.update(40, [_det(100, 100)])          # beyond max gap -> new identity
    assert e[0] not in a


def test_sustained_presence_rejects_pass_through():
    rows = [{"t": t, "track": "S01", "cx": 0, "cy": 0, "w": 40, "h": 40, "yaw": 0, "pitch": .5, "has_pose": True}
            for t in range(0, 100, 2)]
    rows += [{"t": t, "track": "S02", "cx": 0, "cy": 0, "w": 40, "h": 40, "yaw": 0, "pitch": .5, "has_pose": True}
             for t in (10, 12, 14)]
    att = vision.build_attendance(pd.DataFrame(rows), 100, 2, 0.70)
    st = dict(zip(att["Student ID"], att["Status"]))
    assert st["S01"] == "Present" and st["S02"] == "Absent"
    assert list(att.columns[:4]) == ["Student ID", "First Seen", "Last Seen", "Total Duration"]


def test_attendance_boundary_is_strictly_greater():
    rows = [{"t": t, "track": "S01", "cx": 0, "cy": 0, "w": 40, "h": 40, "yaw": 0, "pitch": .5, "has_pose": True}
            for t in range(0, 70)]
    att = vision.build_attendance(pd.DataFrame(rows), 100, 1, 0.70)
    assert att.loc[0, "Status"] == "Absent"       # exactly 70% is not "> 70%"


def test_roster_placeholders():
    att = vision.build_attendance(pd.DataFrame(columns=vision.OBS_COLUMNS), 100, 2, 0.7, expected_roster=3)
    assert len(att) == 3 and (att["Status"] == "Absent").all()


def test_attention_penalises_turned_heads():
    base = dict(t=0, cx=0, cy=0, w=40, h=40, pitch=0.55, has_pose=True)
    obs = pd.DataFrame([dict(base, track="A", yaw=0.0), dict(base, track="B", yaw=0.9)])
    out = vision.score_attention(obs)
    assert out.loc[0, "attention"] > 0.9 > 0.5 > out.loc[1, "attention"]


def _synthetic_audio(sr=16000):
    rng = np.random.default_rng(0)
    t = np.arange(sr) / sr
    silence = 0.0005 * rng.standard_normal(sr * 6)
    n = np.arange(sr * 6) / sr
    voiced = sum(np.sin(2 * np.pi * 140 * k * n) / np.sqrt(k) for k in range(1, 15))
    voiced = 0.15 * voiced * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * np.arange(sr * 6) / sr))
    noise = 0.6 * rng.standard_normal(sr * 6)
    return np.concatenate([silence, voiced, noise, silence]).astype(np.float32)


def test_audio_classification_states():
    df = audio.analyze_audio(_synthetic_audio())
    assert len(df) == 24
    assert (df.loc[1:4, "label"] == "Silence").all()
    assert (df.loc[7:10, "label"].isin(["Instructor Lecture", "Student Discussion"])).all()
    assert (df.loc[13:16, "label"] == "High Ambient Distraction").all()
    assert df["interaction"].between(0, 1).all()


def test_fusion_range_and_alignment():
    v, a = demo.make_demo_vision(), demo.make_demo_audio()
    f = fusion.build_fusion(v, a, len(a), step_s=5)
    assert f["engagement"].between(0, 100).all()
    assert len(f) == 40 * 60 // 5
    q = f.set_index("minute")["engagement"]
    assert q.loc[16:19].mean() < q.loc[23:26].mean() - 15      # dip in script vs. discussion peak


def test_fusion_single_modality():
    a = demo.make_demo_audio()
    f = fusion.build_fusion(None, a, len(a), step_s=10)
    assert f["visual"].isna().all() and f["engagement"].notna().all()


def test_advisor_flags_low_engagement_and_cooldown():
    v, a = demo.make_demo_vision(), demo.make_demo_audio()
    f = fusion.build_fusion(v, a, len(a), step_s=5)
    adv = advisor.generate_advisories(f, advisor.AdvisorConfig())
    assert len(adv) > 0
    assert {"low_engagement"} & set(adv["rule"])
    for rule, g in adv.groupby("rule"):
        assert (g["time_s"].diff().dropna() >= 600).all()


def test_report_outputs():
    v, a = demo.make_demo_vision(), demo.make_demo_audio()
    f = fusion.build_fusion(v, a, len(a), step_s=5)
    adv = advisor.generate_advisories(f, advisor.AdvisorConfig())
    att = vision.build_attendance(v.obs, v.duration_s, v.sample_dt, 0.7)
    k = reporting.compute_kpis(att, f, a, adv, v.duration_s)
    assert 0 <= k["attendance_rate"] <= 100 and k["transient_tracks"] == 2
    md = reporting.build_markdown_report(k, att, adv)
    assert "Teaching advisories" in md
    assert len(reporting.build_zip(md, reporting.kpis_json(k, adv), att, f, a, adv)) > 500


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
