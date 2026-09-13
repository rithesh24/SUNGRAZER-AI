"""Load LASCO level-0.5 FITS files into a validated scientific data model.

``load_science_image()`` is the single entry point every pipeline stage uses
to read a raw frame; it validates required metadata and raises
``FitsLoadError`` for anything unusable, so callers never have to guess
whether a frame is safe to process.

Header facts verified against real level-0.5 files (2026-09-13, see
docs/progress.md): dates are '/'-separated (``DATE-OBS='2024/01/01'``,
``TIME-OBS='00:06:06.030'``); the camera lives in ``DETECTOR`` (``C2``/``C3``)
while ``INSTRUME`` is always ``LASCO``; sun-center is ``CRPIX1/2`` (FITS
1-based pixel convention) with plate scale ``CDELT1/2`` in arcsec/pixel and
roll ``CROTA2`` in degrees; pixel data is a single ``int16`` primary HDU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)


class FitsLoadError(ValueError):
    """Raised when a FITS file is unreadable or lacks required metadata."""


def parse_date_obs(date_obs: str | None) -> datetime | None:
    """Parse a LASCO DATE-OBS string ("2024/01/01T00:06:06.030") to UTC.

    Level-0.5 headers use '/'-separated dates; seconds may or may not carry
    a fractional part. Returns None (never raises) for unparseable values.
    """
    if not date_obs:
        return None
    for fmt in ("%Y/%m/%dT%H:%M:%S.%f", "%Y/%m/%dT%H:%M:%S", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_obs, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    logger.warning("Unparseable DATE-OBS %r", date_obs)
    return None


def combined_date_obs(header: fits.Header) -> str | None:
    """Return the raw 'DATE-OBST TIME-OBS' string, as stored in manifests."""
    date_obs = header.get("DATE-OBS")
    time_obs = header.get("TIME-OBS")
    if date_obs and time_obs:
        return f"{date_obs}T{time_obs}"
    return str(date_obs) if date_obs else None


@dataclass
class ScienceImage:
    """One raw LASCO frame: pixel data plus the metadata the pipeline needs.

    ``sun_center`` keeps the header's FITS 1-based pixel convention
    (CRPIX1, CRPIX2); subtract 1.0 for 0-based array indexing. ``header``
    is the complete original astropy header, so no source metadata is lost
    even if it has no dedicated field here.
    """

    path: Path
    data: np.ndarray                      # 2-D, native dtype (int16 for level 0.5)
    observation_time: datetime            # UTC, required
    telescope: str                        # "SOHO"
    detector: str                         # "C2" | "C3"
    instrument: str                       # "LASCO/C3" — matches images.instrument in the DB
    exposure_seconds: float | None
    filter_name: str | None               # e.g. "Clear"
    polarizer: str | None                 # e.g. "Clear"
    sun_center: tuple[float, float] | None    # (CRPIX1, CRPIX2), FITS 1-based
    plate_scale_arcsec: float | None      # CDELT1, arcsec/pixel
    roll_deg: float | None                # CROTA2
    binning: tuple[int, int]              # (LEBXSUM, LEBYSUM), 1 = unbinned
    header: fits.Header = field(repr=False)

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def height(self) -> int:
        return int(self.data.shape[0])


def _float_or_none(header: fits.Header, key: str) -> float | None:
    value = header.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        logger.warning("Non-numeric header %s=%r", key, value)
        return None


def load_science_image(path: Path | str) -> ScienceImage:
    """Read and validate one level-0.5 FITS file.

    Raises FitsLoadError for: unreadable/corrupt files, no 2-D image data,
    missing/unparseable observation time, or missing detector. Optional
    metadata (WCS, exposure, filter) degrades to None with a log warning.
    """
    path = Path(path)
    try:
        # memmap=False + astype copy: the returned array must own its memory
        # (no dangling file handle — Windows keeps mmap-backed files locked)
        # and be native byte order (FITS is big-endian; OpenCV rejects '>i2').
        with fits.open(path, memmap=False) as hdul:
            hdul.verify("exception")
            image_hdu = next(
                (h for h in hdul
                 if h.data is not None and getattr(h.data, "ndim", 0) == 2),
                None,
            )
            if image_hdu is None:
                raise FitsLoadError(f"{path.name}: no 2-D image data in any HDU")
            header = image_hdu.header
            data = image_hdu.data.astype(
                image_hdu.data.dtype.newbyteorder("="), copy=True
            )
    except FitsLoadError:
        raise
    except Exception as exc:
        raise FitsLoadError(f"{path.name}: unreadable FITS file: {exc}") from exc

    observation_time = parse_date_obs(combined_date_obs(header))
    if observation_time is None:
        raise FitsLoadError(f"{path.name}: missing or unparseable DATE-OBS/TIME-OBS")

    detector = str(header.get("DETECTOR", "")).strip()
    if not detector:
        raise FitsLoadError(f"{path.name}: missing DETECTOR")
    instrume = str(header.get("INSTRUME", "LASCO")).strip() or "LASCO"

    crpix1, crpix2 = _float_or_none(header, "CRPIX1"), _float_or_none(header, "CRPIX2")
    sun_center = (crpix1, crpix2) if crpix1 is not None and crpix2 is not None else None
    if sun_center is None:
        logger.warning("%s: no CRPIX sun-center in header", path.name)

    return ScienceImage(
        path=path,
        data=data,
        observation_time=observation_time,
        telescope=str(header.get("TELESCOP", "")).strip(),
        detector=detector,
        instrument=f"{instrume}/{detector.upper()}",
        exposure_seconds=_float_or_none(header, "EXPTIME"),
        filter_name=str(header["FILTER"]).strip() if "FILTER" in header else None,
        polarizer=str(header["POLAR"]).strip() if "POLAR" in header else None,
        sun_center=sun_center,
        plate_scale_arcsec=_float_or_none(header, "CDELT1"),
        roll_deg=_float_or_none(header, "CROTA2"),
        binning=(int(header.get("LEBXSUM", 1)), int(header.get("LEBYSUM", 1))),
        header=header,
    )
