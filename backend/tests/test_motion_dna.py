"""Offline unit tests for pipeline.motion_dna (no files, no database)."""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.motion_dna import (  # noqa: E402
    MotionDnaConfig,
    compute_features,
    validate_features,
)

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
CADENCE_S = 720  # 12 min
SEQ = [f"f{i:03d}" for i in range(20)]


def make_track(positions, fluxes=None, areas=None, frame_indices=None):
    n = len(positions)
    idx = frame_indices or list(range(n))
    return {
        "track_id": "t0",
        "frames": [SEQ[i] for i in idx],
        "timestamps": [(T0 + timedelta(seconds=CADENCE_S * i)).isoformat()
                       for i in idx],
        "positions": [list(p) for p in positions],
        "fluxes": fluxes or [100.0] * n,
        "areas": areas or [4] * n,
    }


def test_constant_velocity_track():
    # 5 px/frame in +x: straight, smooth, zero acceleration.
    track = make_track([(100 + 5 * i, 200) for i in range(6)])
    f = compute_features(track, SEQ, sun_center=(512.0, 512.0))
    assert f["n_frames"] == 6
    assert f["frame_coverage"] == 1.0
    assert f["duration_s"] == 5 * CADENCE_S
    assert f["displacement_px"] == 25.0
    assert f["path_length_px"] == 25.0
    assert abs(f["speed_mean_px_s"] - 5 / CADENCE_S) < 1e-6  # stored 6-dp
    assert f["direction_deg"] == 0.0
    assert f["direction_consistency"] == 1.0
    assert f["accel_mean_px_s2"] == 0.0
    assert f["curvature_rad_px"] == 0.0
    assert f["linear_rms_px"] < 1e-6
    assert validate_features(f) == []


def test_static_track_zero_motion():
    f = compute_features(make_track([(300, 300)] * 5), SEQ, (512.0, 512.0))
    assert f["speed_mean_px_s"] == 0.0
    assert f["displacement_px"] == 0.0
    assert f["direction_deg"] is None            # undefined, not fabricated
    assert f["direction_consistency"] is None    # no moving steps
    assert f["radial_speed_px_s"] == 0.0
    assert validate_features(f) == []


def test_right_angle_turn_has_curvature():
    positions = [(100 + 5 * i, 200) for i in range(4)]
    positions += [(115, 200 + 5 * i) for i in range(1, 4)]
    f = compute_features(make_track(positions), SEQ, (512.0, 512.0))
    # One 90-degree turn over a 30 px path.
    assert abs(f["curvature_rad_px"] - (math.pi / 2) / 30) < 1e-6
    assert f["direction_consistency"] < 1.0
    assert f["linear_rms_px"] > 1.0
    assert validate_features(f) == []


def test_missed_frame_lowers_coverage():
    f = compute_features(
        make_track([(100, 100), (105, 100), (115, 100), (120, 100)],
                   frame_indices=[0, 1, 3, 4]), SEQ, None)
    assert f["frame_coverage"] == 0.8  # 4 of 5 spanned frames


def test_brightening_flux_positive_slope():
    fluxes = [100.0 + 20.0 * i for i in range(5)]
    f = compute_features(make_track([(100 + i, 100) for i in range(5)],
                                    fluxes=fluxes), SEQ, None)
    # 20 flux/frame = 100/hour on mean 140 -> ~0.714 frac/h.
    assert abs(f["flux_slope_frac_h"] - (20 * 3600 / CADENCE_S) / 140.0) < 1e-3
    assert f["flux_cv"] > 0.0


def test_inbound_track_negative_radial_speed():
    # Moving straight toward the sun center.
    f = compute_features(make_track([(212 + 50 * i, 512) for i in range(5)]),
                         SEQ, (512.0, 512.0))
    assert f["radial_speed_px_s"] < 0
    assert f["r_min_px"] == 100.0
    assert f["r_max_px"] == 300.0


def test_no_sun_center_radial_features_null():
    f = compute_features(make_track([(100 + i, 100) for i in range(5)]),
                         SEQ, None)
    assert f["r_min_px"] is None and f["radial_speed_px_s"] is None
    assert validate_features(f) == []


def test_short_track_higher_order_features_null():
    f = compute_features(make_track([(100, 100), (105, 100)]), SEQ, None)
    assert f["accel_mean_px_s2"] is None
    assert f["curvature_rad_px"] is None
    assert f["linear_rms_px"] is None
    assert f["speed_mean_px_s"] > 0  # first-order still computed
    assert validate_features(f) == []


def test_validate_catches_bad_values():
    f = compute_features(make_track([(100 + i, 100) for i in range(5)]),
                         SEQ, (512.0, 512.0))
    f["frame_coverage"] = 1.5
    f["speed_mean_px_s"] = float("nan")
    problems = validate_features(f)
    assert any("frame_coverage" in p for p in problems)
    assert any("speed_mean_px_s" in p for p in problems)


def test_config_hash_stable():
    assert MotionDnaConfig().config_hash() == MotionDnaConfig().config_hash()


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
