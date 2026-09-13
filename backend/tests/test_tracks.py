"""Offline unit tests for pipeline.tracks (no files, no database)."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.tracks import TracksConfig, build_tracks, track_to_dict  # noqa: E402

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
CADENCE = timedelta(minutes=12)


def make_frames(n: int) -> tuple[list[str], dict[str, datetime]]:
    names = [f"f{i:03d}" for i in range(n)]
    return names, {name: T0 + i * CADENCE for i, name in enumerate(names)}


def det(x: float, y: float, snr: float = 10.0) -> dict:
    return {"x": x, "y": y, "bbox": [int(x) - 1, int(y) - 1, 3, 3],
            "area": 4, "peak": snr, "flux": snr * 4, "snr": snr}


def test_linear_mover_single_track():
    names, times = make_frames(8)
    dets = {name: [det(100 + 4 * i, 200 - 3 * i)] for i, name in enumerate(names)}
    tracks = build_tracks(names, dets, times, TracksConfig())
    assert len(tracks) == 1
    assert len(tracks[0].frames) == 8
    record = track_to_dict(tracks[0], "t0")
    assert record["displacement_px"] == 35.0  # 7 steps of hypot(4,3)=5
    assert record["mean_residual_px"] < 1.0


def test_static_source_tracked_too():
    # Stars/slow drifters must be kept (hard-negative catalog), not dropped.
    names, times = make_frames(6)
    dets = {name: [det(500, 500)] for name in names}
    tracks = build_tracks(names, dets, times, TracksConfig())
    assert len(tracks) == 1
    assert track_to_dict(tracks[0], "t0")["displacement_px"] == 0.0


def test_gap_coasting_keeps_track_alive():
    names, times = make_frames(9)
    dets = {name: [det(100 + 5 * i, 300)] for i, name in enumerate(names)}
    dets[names[4]] = []  # one missed detection mid-track
    tracks = build_tracks(names, dets, times, TracksConfig(max_missed_frames=2))
    assert len(tracks) == 1
    assert len(tracks[0].frames) == 8  # all but the missed frame
    # Velocity prediction bridged the 2-frame time gap (10 px jump > gate
    # from a static prediction, fine from a moving one).
    assert tracks[0].frames == [n for n in names if n != names[4]]


def test_long_dropout_terminates_track():
    names, times = make_frames(10)
    dets = {name: [] for name in names}
    for i in (0, 1, 2):
        dets[names[i]] = [det(100 + 5 * i, 300)]
    for i in (7, 8, 9):
        dets[names[i]] = [det(100 + 5 * i, 300)]
    tracks = build_tracks(names, dets, times, TracksConfig(max_missed_frames=2))
    assert len(tracks) == 2  # 4-frame dropout > max_missed_frames splits it
    assert sorted(len(t.frames) for t in tracks) == [3, 3]


def test_two_parallel_movers_stay_separate():
    names, times = make_frames(6)
    dets = {name: [det(100 + 4 * i, 200), det(100 + 4 * i, 400)]
            for i, name in enumerate(names)}
    tracks = build_tracks(names, dets, times, TracksConfig())
    assert len(tracks) == 2
    assert all(len(t.frames) == 6 for t in tracks)
    ys = sorted(t.ys[0] for t in tracks)
    assert ys == [200.0, 400.0]
    assert all(len(set(t.ys)) == 1 for t in tracks)  # no identity swap


def test_fast_mover_beyond_gate_not_linked():
    names, times = make_frames(5)
    dets = {name: [det(100 + 50 * i, 300)] for i, name in enumerate(names)}
    tracks = build_tracks(names, dets, times, TracksConfig(max_link_px=10.0))
    # 50 px/frame exceeds the gate from a static first prediction; every
    # detection opens its own track (documented failure mode).
    assert len(tracks) == 5


def test_greedy_prefers_nearest():
    names, times = make_frames(2)
    dets = {names[0]: [det(100, 100)],
            names[1]: [det(108, 100), det(101, 100)]}
    tracks = build_tracks(names, dets, times, TracksConfig())
    two_frame = [t for t in tracks if len(t.frames) == 2]
    assert len(two_frame) == 1
    assert two_frame[0].xs == [100.0, 101.0]  # nearest won, 108 seeded new


def test_min_track_frames_is_callers_filter():
    names, times = make_frames(6)
    dets = {name: [det(100 + 4 * i, 200)] for i, name in enumerate(names)}
    dets[names[0]].append(det(900, 900))  # single-frame cosmic ray
    config = TracksConfig()
    tracks = build_tracks(names, dets, times, config)
    assert len(tracks) == 2
    kept = [t for t in tracks if len(t.frames) >= config.min_track_frames]
    assert len(kept) == 1 and len(kept[0].frames) == 6


def test_config_hash_stable_and_sensitive():
    assert TracksConfig().config_hash() == TracksConfig().config_hash()
    assert TracksConfig().config_hash() != TracksConfig(max_link_px=12.0).config_hash()


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
