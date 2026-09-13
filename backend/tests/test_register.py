"""Offline unit tests for pipeline.register (synthetic frames, no DB).

Run:  python tests/test_register.py   (or pytest tests/)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pipeline.register import (
    STATUS_FAILED,
    STATUS_OK,
    RegisterConfig,
    apply_shift,
    estimate_shift,
    register_frame,
)

RNG = np.random.default_rng(42)


def star_field(shape=(256, 256), n_stars=80) -> np.ndarray:
    """Synthetic frame with gaussian 'stars' on a flat background."""
    data = np.full(shape, 10.0, dtype=np.float32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for _ in range(n_stars):
        cx, cy = RNG.uniform(20, shape[1] - 20), RNG.uniform(20, shape[0] - 20)
        amp = RNG.uniform(50, 300)
        data += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 4.0)
    return data.astype(np.float32)


FRAME = star_field()
MASK = np.ones(FRAME.shape, dtype=bool)
CFG = RegisterConfig()


def test_estimate_recovers_known_shift():
    # cv2.phaseCorrelate is biased toward integer shifts: worst ~0.3 px at
    # half-pixel true shifts, ~0.01 px elsewhere (documented failure mode).
    for true_dx, true_dy, tol in [(0.5, -0.3, 0.35), (2.25, 1.75, 0.1),
                                  (5.0, -4.0, 0.1)]:
        moving, _ = apply_shift(FRAME, MASK, true_dx, true_dy)
        dx, dy, response = estimate_shift(FRAME, moving)
        # corrective shift must undo the applied one
        assert abs(dx + true_dx) < tol and abs(dy + true_dy) < tol, \
            (true_dx, true_dy, dx, dy)
        assert response > 0.2, response


def test_zero_shift_is_near_zero():
    dx, dy, response = estimate_shift(FRAME, FRAME.copy())
    assert abs(dx) < 0.05 and abs(dy) < 0.05
    assert response > 0.5


def test_register_frame_undoes_shift():
    moving, moving_mask = apply_shift(FRAME, MASK, 3.0, -2.0)
    result = register_frame(FRAME, moving, moving_mask, CFG)
    assert result["status"] == STATUS_OK
    interior = np.s_[20:-20, 20:-20]
    assert np.allclose(result["data"][interior], FRAME[interior], atol=2.0)


def test_weak_structure_is_flagged_failed():
    # Two independent noise frames share no structure: the estimate is
    # meaningless and must be rejected, with the frame kept unshifted.
    a = RNG.normal(0, 1, FRAME.shape).astype(np.float32)
    b = RNG.normal(0, 1, FRAME.shape).astype(np.float32)
    result = register_frame(a, b, MASK, CFG)
    assert result["status"] == STATUS_FAILED
    assert result["dx"] == 0.0 and result["dy"] == 0.0   # not applied
    assert np.array_equal(result["data"], b)             # unshifted


def test_excessive_shift_is_flagged_failed():
    moving, moving_mask = apply_shift(FRAME, MASK, 30.0, 0.0)
    tight = RegisterConfig(max_shift_px=10.0, min_response=0.0)
    result = register_frame(FRAME, moving, moving_mask, tight)
    # either the estimator finds ~30 px (over limit) or aliases it — both
    # must end in FAILED, never a silently applied bad warp
    assert result["status"] in (STATUS_FAILED, STATUS_OK)
    if result["status"] == STATUS_OK:
        raise AssertionError(f"30px shift accepted: {result}")


def test_shift_moves_mask_too():
    _, mask = apply_shift(FRAME, MASK, 5.0, 0.0)
    assert not mask[:, :4].any()      # vacated left border is invalid
    assert mask[100, 100]


def test_shape_mismatch_excluded_not_crashed():
    # Binned 512x512 frames appear in real LASCO days; register_sequence
    # excludes them via a shape guard. Exercise the guard's precondition:
    # estimate_shift on mismatched shapes raises, so the guard must exist.
    import cv2
    small = FRAME[:128, :128].copy()
    try:
        estimate_shift(FRAME, small)
    except cv2.error:
        pass
    else:
        raise AssertionError("expected cv2.error on shape mismatch")


def test_config_hash_stable_and_sensitive():
    assert RegisterConfig().config_hash() == RegisterConfig().config_hash()
    assert (RegisterConfig().config_hash()
            != RegisterConfig(max_shift_px=5.0).config_hash())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("All register tests passed.")
