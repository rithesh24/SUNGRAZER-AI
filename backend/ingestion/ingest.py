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
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

from ingestion import archive_client, config
from ingestion.manifest import Manifest, STATUS_COMPLETE, STATUS_FAILED
from pipeline.fits_io import combined_date_obs, load_science_image

logger = logging.getLogger(__name__)


def validate_fits(path: Path) -> str | None:
    """Validate the file via the shared loader; return the raw DATE-OBS string.

    Delegates to pipeline.fits_io.load_science_image, which raises
    FitsLoadError for unreadable files, missing 2-D data, or a missing
    observation time — the last is stricter than the original prototype,
    which tolerated absent DATE-OBS; frames without a timestamp cannot be
    sequenced and are now recorded as failed ingestions.
    """
    image = load_science_image(path)
    return combined_date_obs(image.header)


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
    downloaded = failed = 0

    todo = [f for f in files if not manifest.is_complete(f)]
    skipped = len(files) - len(todo)
    day_base = archive_client.day_url(base_url, obs_date, camera)

    # Downloads run in parallel (SDAC stalls the first request per file while
    # it warms the cache, so serial ingest pays ~30s per frame; overlapping
    # the stalls recovers most of it). Validation and manifest writes stay in
    # this thread — Manifest is a plain JSON file, not thread-safe.
    with ThreadPoolExecutor(max_workers=config.DOWNLOAD_WORKERS) as pool:
        futures = [(f, pool.submit(archive_client.download_file,
                                   day_base + f, day_dir / f)) for f in todo]
        for filename, future in futures:
            url = day_base + filename
            dest = day_dir / filename
            try:
                sha256 = future.result()
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
    # A few flaky frames must not abort the day (sequences tolerate gaps and
    # the manifest retries them next run); fail only when nothing came down.
    if summary["failed"] and not (summary["downloaded"] + summary["skipped"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
