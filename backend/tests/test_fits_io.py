"""Unit tests for pipeline.fits_io (synthetic FITS files, offline).

Run:  python tests/test_fits_io.py   (or pytest tests/)

If sample data from the ingestion smoke tests exists on disk
(data/raw/soho/lasco/240101/c3), one extra test validates against a real
level-0.5 file; otherwise it is skipped.
"""

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from astropy.io import fits

from pipeline.fits_io import FitsLoadError, load_science_image

REAL_SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data/raw/soho/lasco/240101/c3"


def write_fits(directory: Path, name: str = "test.fts", *,
               data=None, drop: tuple[str, ...] = ()) -> Path:
    """Write a minimal LASCO-like FITS file; ``drop`` removes header keys."""
    if data is None:
        data = np.arange(64, dtype=np.int16).reshape(8, 8)
    header = fits.Header({
        "DATE-OBS": "2024/01/01", "TIME-OBS": "00:06:06.030",
        "TELESCOP": "SOHO", "INSTRUME": "LASCO", "DETECTOR": "C3",
        "EXPTIME": 17.5986, "FILTER": "Clear", "POLAR": "Clear",
        "CRPIX1": 519.2, "CRPIX2": 533.5, "CDELT1": 56.0, "CDELT2": 56.0,
        "CROTA2": 173.132, "LEBXSUM": 1, "LEBYSUM": 1,
    })
    for key in drop:
        del header[key]
    path = directory / name
    fits.PrimaryHDU(data=data, header=header).writeto(path)
    return path


def test_load_valid_file():
    with tempfile.TemporaryDirectory() as tmp:
        img = load_science_image(write_fits(Path(tmp)))
        assert img.observation_time == datetime(2024, 1, 1, 0, 6, 6, 30000,
                                                tzinfo=timezone.utc)
        assert img.telescope == "SOHO"
        assert img.detector == "C3"
        assert img.instrument == "LASCO/C3"
        assert img.width == 8 and img.height == 8
        assert img.data.dtype == np.int16 and img.data.dtype.isnative
        assert abs(img.exposure_seconds - 17.5986) < 1e-6
        assert img.filter_name == "Clear" and img.polarizer == "Clear"
        assert img.sun_center == (519.2, 533.5)
        assert img.plate_scale_arcsec == 56.0
        assert abs(img.roll_deg - 173.132) < 1e-6
        assert img.binning == (1, 1)
        assert img.header["INSTRUME"] == "LASCO"  # full header preserved


def test_missing_observation_time_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_fits(Path(tmp), drop=("DATE-OBS", "TIME-OBS"))
        try:
            load_science_image(path)
        except FitsLoadError as exc:
            assert "DATE-OBS" in str(exc)
        else:
            raise AssertionError("expected FitsLoadError")


def test_missing_detector_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_fits(Path(tmp), drop=("DETECTOR",))
        try:
            load_science_image(path)
        except FitsLoadError as exc:
            assert "DETECTOR" in str(exc)
        else:
            raise AssertionError("expected FitsLoadError")


def test_no_2d_data_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_fits(Path(tmp), data=np.arange(8, dtype=np.int16))
        try:
            load_science_image(path)
        except FitsLoadError as exc:
            assert "2-D" in str(exc)
        else:
            raise AssertionError("expected FitsLoadError")


def test_garbage_file_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "garbage.fts"
        path.write_bytes(b"this is not a FITS file" * 100)
        try:
            load_science_image(path)
        except FitsLoadError as exc:
            assert "unreadable" in str(exc)
        else:
            raise AssertionError("expected FitsLoadError")


def test_missing_wcs_degrades_to_none():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_fits(Path(tmp), drop=("CRPIX1", "CRPIX2", "CDELT1", "CROTA2"))
        img = load_science_image(path)
        assert img.sun_center is None
        assert img.plate_scale_arcsec is None
        assert img.roll_deg is None
        assert img.observation_time is not None  # still loads


def test_real_sample_file():
    samples = sorted(REAL_SAMPLE_DIR.glob("*.fts")) if REAL_SAMPLE_DIR.exists() else []
    if not samples:
        print("SKIP test_real_sample_file (no sample data on disk)")
        return
    img = load_science_image(samples[0])
    assert img.instrument == "LASCO/C3"
    assert img.width == 1024 and img.height == 1024
    assert img.observation_time.year == 2024
    assert img.sun_center is not None
    assert img.exposure_seconds is not None and img.exposure_seconds > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("All fits_io tests passed.")
