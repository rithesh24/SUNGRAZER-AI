"""Scientific preprocessing: raw LASCO frame -> normalized frame + validity mask.

Usage (batch over one DB sequence):
    python -m pipeline.preprocess --sequence-id N [--data-root PATH]

Baseline operations, in order (techspec section 9):

1. exposure normalization  — int16 DN -> float32 DN/second, so frames with
   different exposure times are photometrically comparable
2. invalid-pixel handling  — non-finite or <= ``invalid_below`` DN pixels
   (LASCO missing telemetry blocks are zero) are excluded via the mask and
   zeroed in the data
3. field masking           — annulus between occulter and outer field of
   view around the header sun-center, plus an image-border margin
4. optional noise filter   — median filter, off by default: a faint comet
   spans only a few pixels, so the default pipeline must not smooth it away
5. optional stretch        — none (default) | log | asinh dynamic-range
   transform; 'none' preserves linear photometry for differencing
6. quality assessment      — invalid fraction, exposure sanity, usable flag

Determinism/provenance: every operation is driven by ``PreprocessConfig``
(version ``prep_v1``); outputs carry the full config dict and its SHA-256
hash, and the batch CLI writes both into a per-sequence JSON manifest next
to the ``.npz`` outputs. Raw files are never modified (raw/ vs processed/).

Instrument geometry (verified 2026-09-13 against NASA SPASE/data.gov LASCO
records and Battams & Knight 2016, arXiv:1611.02279): C2 images 1.5-6 R_sun
at ~12 arcsec/px; C3 images 3.7-30 R_sun at 56 arcsec/px. Default mask radii
sit slightly outside the physical occulter because its edge is surrounded by
diffraction/stray light. All radii are configurable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from pipeline.fileio import atomic_write_bytes_via, atomic_write_text
from pipeline.fits_io import ScienceImage, load_science_image

logger = logging.getLogger(__name__)

PREPROCESS_VERSION = "prep_v1"

# Verified instrument FOV (R_sun); occulter defaults add margin for the
# diffraction ring around the physical occulter edge.
DETECTOR_DEFAULTS = {
    "C2": {"occulter_rsun": 2.2, "fov_rsun": 6.0},
    "C3": {"occulter_rsun": 4.0, "fov_rsun": 30.0},
}

# Apparent solar radius from SOHO's L1 orbit (~0.99 AU); mask radii shift by
# well under a pixel across the true seasonal range, so a constant suffices.
DEFAULT_RSUN_ARCSEC = 965.0


@dataclass(frozen=True)
class PreprocessConfig:
    """Deterministic, hashable configuration for one preprocessing run."""

    version: str = PREPROCESS_VERSION
    rsun_arcsec: float = DEFAULT_RSUN_ARCSEC
    occulter_rsun: float = 4.0          # per-detector; see for_detector()
    fov_rsun: float = 30.0
    border_px: int = 8                  # CCD edge artifacts
    invalid_below: float = 0.0          # DN <= this are invalid (missing blocks)
    median_filter_px: int = 0           # 0 = off; a faint comet is a few px
    stretch: str = "none"               # none | log | asinh
    max_invalid_fraction: float = 0.5   # QA: above this the frame is unusable

    @classmethod
    def for_detector(cls, detector: str, **overrides) -> "PreprocessConfig":
        defaults = DETECTOR_DEFAULTS.get(detector.upper(), {})
        return cls(**{**defaults, **overrides})

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class PreprocessedFrame:
    """Normalized frame + validity mask + QA + provenance."""

    data: np.ndarray                    # float32, invalid pixels zeroed
    valid_mask: np.ndarray              # bool, True = scientifically usable pixel
    qa: dict
    config: PreprocessConfig
    source_path: Path
    observation_time: datetime
    instrument: str


def build_field_mask(shape: tuple[int, int],
                     sun_center: tuple[float, float] | None,
                     plate_scale_arcsec: float | None,
                     config: PreprocessConfig) -> np.ndarray:
    """Boolean mask of usable field pixels (True = keep).

    Masks the occulter disk, the region outside the outer FOV, and an image
    border. Without a sun-center or plate scale only the border is masked
    (the caller's QA records the degradation).
    """
    height, width = shape
    mask = np.ones(shape, dtype=bool)
    b = config.border_px
    if b > 0:
        mask[:b, :] = mask[-b:, :] = False
        mask[:, :b] = mask[:, -b:] = False
    if sun_center is None or not plate_scale_arcsec:
        return mask
    # FITS CRPIX is 1-based -> subtract 1 for array coordinates.
    cx, cy = sun_center[0] - 1.0, sun_center[1] - 1.0
    rsun_px = config.rsun_arcsec / plate_scale_arcsec
    yy, xx = np.mgrid[0:height, 0:width]
    r = np.hypot(xx - cx, yy - cy)
    mask &= (r >= config.occulter_rsun * rsun_px)
    mask &= (r <= config.fov_rsun * rsun_px)
    return mask


def apply_stretch(data: np.ndarray, stretch: str) -> np.ndarray:
    """Optional dynamic-range transform on non-negative DN/s data."""
    if stretch == "none":
        return data
    if stretch == "log":
        return np.log1p(np.clip(data, 0, None)).astype(np.float32)
    if stretch == "asinh":
        return np.arcsinh(data).astype(np.float32)
    raise ValueError(f"Unknown stretch {stretch!r}")


def preprocess(image: ScienceImage, config: PreprocessConfig) -> PreprocessedFrame:
    """Run the deterministic preprocessing chain on one loaded frame."""
    raw = image.data.astype(np.float32, copy=True)

    invalid = ~np.isfinite(raw) | (raw <= config.invalid_below)
    field_mask = build_field_mask(raw.shape, image.sun_center,
                                  image.plate_scale_arcsec, config)
    valid_mask = field_mask & ~invalid

    exposure_ok = image.exposure_seconds is not None and image.exposure_seconds > 0
    data = raw / image.exposure_seconds if exposure_ok else raw
    if config.median_filter_px > 1:
        data = median_filter(data, size=config.median_filter_px)
    data = apply_stretch(data, config.stretch)
    data = np.where(valid_mask, data, 0.0).astype(np.float32)

    field_px = int(field_mask.sum())
    invalid_fraction = (float((invalid & field_mask).sum()) / field_px
                        if field_px else 1.0)
    qa = {
        "exposure_ok": exposure_ok,
        "sun_center_missing": image.sun_center is None,
        "invalid_fraction": round(invalid_fraction, 6),
        "field_pixels": field_px,
        "usable": exposure_ok and field_px > 0
                  and invalid_fraction <= config.max_invalid_fraction,
    }
    if not qa["usable"]:
        logger.warning("Frame %s failed QA: %s", image.path.name, qa)

    return PreprocessedFrame(
        data=data, valid_mask=valid_mask, qa=qa, config=config,
        source_path=image.path, observation_time=image.observation_time,
        instrument=image.instrument,
    )


def save_frame(frame: PreprocessedFrame, dest: Path) -> None:
    """Write data + mask as compressed npz (atomic, lock-retrying)."""
    atomic_write_bytes_via(
        dest, lambda fh: np.savez_compressed(fh, data=frame.data,
                                             valid_mask=frame.valid_mask))


def preprocess_sequence(sequence_id: int, data_root: Path) -> dict:
    """Preprocess every complete image of one DB sequence to disk.

    Writes ``<data_root>/processed/normalized/seq_<id>/<image-stem>.npz``
    plus ``preprocess_manifest.json`` recording the config (dict + hash) and
    per-frame QA/provenance. Idempotent: frames already listed in the
    manifest with the same config hash are skipped.
    """
    # Local import so the pure functions above stay usable without DB deps.
    from sqlalchemy import select

    from db.models import Image, ImageSequence
    from db.session import SessionLocal

    with SessionLocal() as session:
        sequence = session.get(ImageSequence, sequence_id)
        if sequence is None:
            raise ValueError(f"sequence {sequence_id} does not exist")
        images = session.scalars(
            select(Image)
            .where(Image.sequence_id == sequence_id,
                   Image.ingestion_status == "complete")
            .order_by(Image.observation_time)
        ).all()

    detector = sequence.instrument.split("/")[-1]
    config = PreprocessConfig.for_detector(detector)
    out_dir = data_root / "processed" / "normalized" / f"seq_{sequence_id}"
    manifest_path = out_dir / "preprocess_manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {"frames": {}})
    if manifest.get("config_hash") not in (None, config.config_hash()):
        # ponytail: config changed -> redo everything; per-config output dirs
        # can come later if we ever need to keep several variants side by side
        logger.info("Config hash changed; reprocessing all frames")
        manifest = {"frames": {}}
    manifest["config"] = config.to_dict()
    manifest["config_hash"] = config.config_hash()
    manifest["sequence_id"] = sequence_id

    processed = skipped = failed = 0
    for db_image in images:
        name = Path(db_image.local_path).stem
        if name in manifest["frames"] and manifest["frames"][name].get("ok"):
            skipped += 1
            continue
        source = data_root / db_image.local_path
        try:
            frame = preprocess(load_science_image(source), config)
            save_frame(frame, out_dir / f"{name}.npz")
            manifest["frames"][name] = {
                "ok": True,
                "source": db_image.local_path,
                "image_id": db_image.id,
                "qa": frame.qa,
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }
            processed += 1
        except Exception as exc:  # one bad frame must not kill the sequence
            failed += 1
            logger.error("Preprocess failed for %s: %s", source, exc)
            manifest["frames"][name] = {"ok": False, "error": str(exc),
                                        "source": db_image.local_path}
        atomic_write_text(manifest_path,
                          json.dumps(manifest, indent=2, sort_keys=True))

    summary = {"sequence_id": sequence_id, "frames": len(images),
               "processed": processed, "skipped": skipped, "failed": failed}
    logger.info("Preprocess summary: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess all frames of one image sequence")
    parser.add_argument("--sequence-id", type=int, required=True)
    parser.add_argument("--data-root", default=os.environ.get("SOHO_DATA_ROOT", "data"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    summary = preprocess_sequence(args.sequence_id, Path(args.data_root))
    # A few bad frames must not abort the sequence; fail only when there were
    # failures and nothing succeeded (skipped = already done, counts as ok).
    if summary["failed"] and not (summary["processed"] + summary["skipped"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
