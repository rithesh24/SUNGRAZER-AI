"""Offline unit tests for pipeline.motion (synthetic sequences, no files).

Run:  python tests/test_motion.py   (or pytest tests/)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pipeline.motion import (
    MotionConfig,
    extract_detections,
    robust_sigma,
    run_motion,
    temporal_background,
)

RNG = np.random.default_rng(7)
SHAPE = (128, 128)
MASK = np.ones(SHAPE, dtype=bool)


def gaussian(x, y, amp, sigma=1.5):
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    return amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))


def make_sequence(n=7, noise=1.0, star_amp=100.0, blob_amp=15.0):
    """Static star at (30,30); moving blob from (60,90) at (+5,-5)/frame."""
    frames, positions = [], []
    for i in range(n):
        frame = RNG.normal(50.0, noise, SHAPE).astype(np.float32)
        frame += gaussian(30, 30, star_amp)
        x, y = 60.0 + 5 * i, 90.0 - 5 * i
        frame += gaussian(x, y, blob_amp)
        frames.append(frame.astype(np.float32))
        positions.append((x, y))
    return frames, positions


NAMES = [f"f{i}" for i in range(7)]
FRAMES, POSITIONS = make_sequence()
MASKS = [MASK] * 7


def hits(detections, x, y, radius=2.5):
    return [d for d in detections if np.hypot(d["x"] - x, d["y"] - y) <= radius]


def test_robust_sigma_matches_normal():
    values = RNG.normal(0, 2.0, 100_000)
    assert abs(robust_sigma(values) - 2.0) < 0.05


def test_temporal_background_excludes_self():
    frames = [np.full(SHAPE, v, dtype=np.float32) for v in (1, 1, 99, 1, 1)]
    bg = temporal_background(frames, 2, window=5)
    assert np.allclose(bg, 1.0)   # the 99-frame is not in its own background


def test_median_detects_mover_not_star():
    result = run_motion(NAMES, FRAMES, MASKS, MotionConfig(method="median"))
    for i, name in enumerate(NAMES):
        detections = result["frames"][name]["detections"]
        x, y = POSITIONS[i]
        assert hits(detections, x, y), f"mover missed in frame {i}"
        assert not hits(detections, 30, 30), f"static star detected in frame {i}"


def test_prev_diff_detects_mover():
    result = run_motion(NAMES, FRAMES, MASKS, MotionConfig(method="prev_diff"))
    assert result["frames"]["f0"]["detections"] == []   # no previous frame
    for i in range(1, 7):
        x, y = POSITIONS[i]
        assert hits(result["frames"][NAMES[i]]["detections"], x, y)


def test_pure_noise_yields_no_detections():
    residual = RNG.normal(0, 1.0, SHAPE).astype(np.float32)
    detections, sigma = extract_detections(residual, MASK, MotionConfig())
    assert sigma > 0
    assert detections == []   # 5-sigma + min_area 2 kills noise


def test_area_filters():
    residual = np.zeros(SHAPE, dtype=np.float32)
    residual[10, 10] = 100.0                 # 1 px: below min_area 2
    residual[40:44, 40:44] = 100.0           # 16 px blob: kept
    residual[80:120, 20:120] = 100.0         # 4000 px: above max_area
    # noise floor so sigma > 0
    residual += RNG.normal(0, 0.5, SHAPE).astype(np.float32)
    detections, _ = extract_detections(residual, MASK, MotionConfig())
    assert len(hits(detections, 41.5, 41.5)) == 1
    assert not hits(detections, 10, 10, radius=1.5)
    assert all(d["area"] <= 2000 for d in detections)


def test_detection_fields_sane():
    result = run_motion(NAMES, FRAMES, MASKS, MotionConfig(method="median"))
    d = hits(result["frames"]["f3"]["detections"], *POSITIONS[3])[0]
    assert d["snr"] >= 5.0
    assert d["flux"] > 0 and d["peak"] > 0
    bx, by, bw, bh = d["bbox"]
    assert bx <= d["x"] <= bx + bw and by <= d["y"] <= by + bh


def test_config_hash_stable_and_sensitive():
    assert MotionConfig().config_hash() == MotionConfig().config_hash()
    assert (MotionConfig().config_hash()
            != MotionConfig(threshold_sigma=4.0).config_hash())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("All motion tests passed.")
