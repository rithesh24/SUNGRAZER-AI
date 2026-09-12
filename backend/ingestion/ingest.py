"""Ingest one day of LASCO level-0.5 imagery.

Usage:
    python -m ingestion.ingest --date 2024-01-01 --camera c3 [--limit N]

Tries the NRL historical archive first, then falls back to the NASA SDAC
recent archive (which only holds roughly the last two weeks). Downloads are
idempotent: files already recorded as complete in the day manifest are
skipped, and every stored file is validated as readable FITS before being
marked complete.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

from astropy.io import fits

from ingestion import archive_client, config
from ingestion.manifest import Manifest, STATUS_COMPLETE, STATUS_FAILED

logger = logging.getLogger(__name__)


def validate_fits(path: Path) -> str | None:
    """Open the file as FITS and return DATE-OBS if present.

    Raises on unreadable/corrupt files. Level-0.5 files carry image data in
    the primary HDU; we require at least a 2-D data array somewhere.
    """
    with fits.open(path) as hdul:
        hdul.verify("exception")
        if not any(hdu.data is not None and getattr(hdu.data, "ndim", 0) >= 2 for hdu in hdul):
            raise ValueError(f"{path.name}: no 2-D image data in any HDU")
        header = hdul[0].header
        date_obs = header.get("DATE-OBS")
        time_obs = header.get("TIME-OBS")
        if date_obs and time_obs:
            return f"{date_obs}T{time_obs}"
        return str(date_obs) if date_obs else None


def ingest_day(obs_date: date, camera: str, limit: int | None = None) -> dict:
    """Download and validate one day/camera. Returns a summary dict."""
    files: list[str] = []
    base_url = None
    for candidate_base in (config.HISTORICAL_BASE_URL, config.RECENT_BASE_URL):
        files = archive_client.list_day_files(obs_date, camera, candidate_base)
        if files:
            base_url = candidate_base
            break
    if not files:
        logger.warning("No data found for %s %s in either archive", obs_date, camera)
        return {"date": str(obs_date), "camera": camera, "listed": 0,
                "downloaded": 0, "skipped": 0, "failed": 0}

    if limit is not None:
        files = files[:limit]

    day_dir = config.RAW_DIR / f"{obs_date:%y%m%d}" / camera
    manifest = Manifest.load(day_dir)
    downloaded = skipped = failed = 0

    for filename in files:
        if manifest.is_complete(filename):
            skipped += 1
            continue
        url = archive_client.day_url(base_url, obs_date, camera) + filename
        dest = day_dir / filename
        try:
            sha256 = archive_client.download_file(url, dest)
            date_obs = validate_fits(dest)
            manifest.record(filename, status=STATUS_COMPLETE, source_url=url,
                            sha256=sha256, size_bytes=dest.stat().st_size,
                            date_obs=date_obs)
            downloaded += 1
            logger.info("Ingested %s (DATE-OBS=%s)", filename, date_obs)
        except Exception as exc:  # record the failure; one bad file must not kill the run
            failed += 1
            logger.error("Failed to ingest %s: %s", url, exc)
            manifest.record(filename, status=STATUS_FAILED, source_url=url, error=str(exc))
            dest.unlink(missing_ok=True)
        manifest.save()  # save per file so an interrupt loses nothing

    summary = {"date": str(obs_date), "camera": camera, "listed": len(files),
               "downloaded": downloaded, "skipped": skipped, "failed": failed}
    logger.info("Ingestion summary: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest one day of LASCO level-0.5 imagery")
    parser.add_argument("--date", required=True, help="Observation date, YYYY-MM-DD")
    parser.add_argument("--camera", default="c3", choices=config.CAMERAS)
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of files to ingest (for testing)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    obs_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    if obs_date > date.today():
        parser.error("--date must not be in the future")

    summary = ingest_day(obs_date, args.camera, limit=args.limit)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
