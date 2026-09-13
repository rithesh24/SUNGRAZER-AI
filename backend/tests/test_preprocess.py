"""Offline unit tests for pipeline.preprocess (no DB, no files).

Run:  python tests/test_preprocess.py   (or pytest tests/)
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from astropy.io import fits

from pipeline.fits_io import ScienceImage
from pipeline.preprocess import (
    PreprocessConfig,
    apply_stretch,
    build_field_mask,
    preprocess,
)

T = datetime(2024, 1, 1, tzinfo=timezone.utc)


def make_image(data, *, sun_center=(50.0, 50.0), plate_scale=56.0,
               exposure=10.0) -> ScienceImage:
    return ScienceImage(
        path=Path("synthetic.fts"), data=data, observation_time=T,
        telescope="SOHO", detector="C3", instrument="LASCO/C3",
        exposure_seconds=exposure, filter_name="Clear", polarizer="Clear",
        sun_center=sun_center, plate_scale_arcsec=plate_scale,
        roll_deg=0.0, binning=(1, 1), header=fits.Header(),
    )


# rsun_arcsec=56.0 with plate_scale=56.0 -> 1 R_sun == 1 pixel, so radii in
# the tests below read directly as pixels.
CFG = PreprocessConfig(rsun_arcsec=56.0, occulter_rsun=10.0, fov_rsun=40.0,
                       border_px=2)


def test_field_mask_geometry():
    mask = build_field_mask((100, 100), (50.0, 50.0), 56.0, CFG)
    assert not mask[49, 49]          # sun center (0-based 49,49): occulted
    assert not mask[49, 55]          # r=6 px < occulter 10 px
    assert mask[49, 69]              # r=20 px: inside annulus
    assert not mask[1, 1]            # border margin
    assert not mask[49, 95]          # r=46 px > fov 40 px


def test_field_mask_without_sun_center_masks_border_only():
    mask = build_field_mask((100, 100), None, None, CFG)
    assert mask[50, 50] and mask[49, 55]      # no occulter masking
    assert not mask[0, 50] and not mask[50, 99]


def test_exposure_normalization():
    data = np.full((100, 100), 100, dtype=np.int16)
    frame = preprocess(make_image(data, exposure=10.0), CFG)
    assert frame.data[frame.valid_mask].max() == np.float32(10.0)  # 100 DN / 10 s
    assert frame.qa["exposure_ok"] and frame.qa["usable"]


def test_invalid_pixels_masked_and_zeroed():
    data = np.full((100, 100), 100, dtype=np.float32)
    data[49, 69] = 0.0        # missing-block pixel inside the annulus
    data[49, 70] = np.nan
    frame = preprocess(make_image(data), CFG)
    assert not frame.valid_mask[49, 69] and not frame.valid_mask[49, 70]
    assert frame.data[49, 69] == 0.0 and frame.data[49, 70] == 0.0
    assert frame.qa["invalid_fraction"] > 0


def test_unusable_when_mostly_invalid():
    data = np.zeros((100, 100), dtype=np.float32)  # everything <= invalid_below
    frame = preprocess(make_image(data), CFG)
    assert frame.qa["invalid_fraction"] == 1.0
    assert not frame.qa["usable"]


def test_missing_exposure_flags_qa():
    data = np.full((100, 100), 100, dtype=np.int16)
    frame = preprocess(make_image(data, exposure=None), CFG)
    assert not frame.qa["exposure_ok"] and not frame.qa["usable"]
    assert frame.data[frame.valid_mask].max() == np.float32(100.0)  # raw passthrough


def test_faint_signal_preserved_by_default_config():
    # A comet-like 2x2 blob 5 DN above background must survive untouched.
    data = np.full((100, 100), 100, dtype=np.float32)
    data[30:32, 30:32] = 105.0
    frame = preprocess(make_image(data, exposure=1.0), CFG)
    assert np.allclose(frame.data[30:32, 30:32], 105.0)
    assert np.allclose(frame.data[35, 35], 100.0)


def test_stretch_variants():
    data = np.array([[0.0, 10.0]], dtype=np.float32)
    assert np.allclose(apply_stretch(data, "none"), data)
    log = apply_stretch(data, "log")
    asinh = apply_stretch(data, "asinh")
    assert log[0, 1] > log[0, 0] and asinh[0, 1] > asinh[0, 0]
    try:
        apply_stretch(data, "sqrt")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown stretch")


def test_config_hash_stable_and_sensitive():
    assert PreprocessConfig().config_hash() == PreprocessConfig().config_hash()
    assert (PreprocessConfig(occulter_rsun=4.0).config_hash()
            != PreprocessConfig(occulter_rsun=4.5).config_hash())


def test_detector_defaults():
    c2, c3 = PreprocessConfig.for_detector("c2"), PreprocessConfig.for_detector("C3")
    assert c2.occulter_rsun == 2.2 and c2.fov_rsun == 6.0
    assert c3.occulter_rsun == 4.0 and c3.fov_rsun == 30.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("All preprocess tests passed.")
