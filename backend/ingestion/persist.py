"""Persist day-manifest ingestion metadata into PostgreSQL.

Usage:
    python -m ingestion.persist [--data-root PATH]

Walks every ``<raw>/<YYMMDD>/<camera>/manifest.json`` under the raw data
tree and upserts one ``images`` row per manifest entry, keyed by the stable
source identifier ``<YYMMDD>/<camera>/<filename>``. Re-running is idempotent:
unchanged entries are skipped, changed entries (e.g. a failed download that
later succeeded) are updated in place. The manifest stays the collector-side
record; PostgreSQL becomes the queryable system of record (claude.md
sections 15 and 23).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from astropy.io import fits
from sqlalchemy.orm import Session

from db.models import Image, Mission
from db.session import SessionLocal
from ingestion import config
from ingestion.manifest import MANIFEST_NAME, Manifest
from pipeline.fits_io import parse_date_obs  # canonical DATE-OBS parser

logger = logging.getLogger(__name__)

MISSION_NAME = "SOHO"
INSTRUMENT_NAME = "LASCO"


def read_dimensions(path: Path) -> tuple[int | None, int | None]:
    """Return (width, height) from the first 2-D HDU, or (None, None)."""
    try:
        with fits.open(path) as hdul:
            for hdu in hdul:
                if hdu.data is not None and getattr(hdu.data, "ndim", 0) >= 2:
                    height, width = hdu.data.shape[-2], hdu.data.shape[-1]
                    return int(width), int(height)
    except Exception as exc:  # metadata-only backfill must survive a bad file
        logger.warning("Could not read dimensions from %s: %s", path, exc)
    return None, None


def get_or_create_mission(session: Session) -> Mission:
    """Return the SOHO/LASCO mission row, creating it on first run."""
    mission = (
        session.query(Mission)
        .filter_by(mission_name=MISSION_NAME, instrument_name=INSTRUMENT_NAME)
        .one_or_none()
    )
    if mission is None:
        mission = Mission(
            mission_name=MISSION_NAME,
            instrument_name=INSTRUMENT_NAME,
            description="SOHO LASCO coronagraph, level-0.5 imagery",
        )
        session.add(mission)
        session.flush()
        logger.info("Created mission %s/%s (id=%s)", MISSION_NAME, INSTRUMENT_NAME, mission.id)
    return mission


def image_row_values(source_identifier: str, day_dir_name: str, camera: str,
                     filename: str, entry: dict, raw_dir: Path) -> dict:
    """Map one manifest entry to images-table column values."""
    local_path = f"raw/soho/lasco/{day_dir_name}/{camera}/{filename}"
    width = height = None
    if entry.get("status") == "complete":
        width, height = read_dimensions(raw_dir / day_dir_name / camera / filename)
    return {
        "source_identifier": source_identifier,
        "source_url": entry.get("source_url", ""),
        "local_path": local_path,
        "observation_time": parse_date_obs(entry.get("date_obs")),
        "instrument": f"{INSTRUMENT_NAME}/{camera.upper()}",
        "width": width,
        "height": height,
        "checksum": entry.get("sha256"),
        "size_bytes": entry.get("size_bytes"),
        "ingestion_status": entry.get("status", "failed"),
        "error_message": entry.get("error"),
    }


def persist_manifests(session: Session, raw_dir: Path) -> dict:
    """Upsert every manifest entry under raw_dir. Returns a summary dict."""
    mission = get_or_create_mission(session)
    created = updated = unchanged = 0

    for manifest_path in sorted(raw_dir.glob(f"*/*/{MANIFEST_NAME}")):
        camera_dir = manifest_path.parent
        day_dir_name = camera_dir.parent.name  # YYMMDD
        camera = camera_dir.name  # c2 | c3
        manifest = Manifest.load(camera_dir)

        for filename, entry in sorted(manifest.entries.items()):
            source_identifier = f"{day_dir_name}/{camera}/{filename}"
            values = image_row_values(
                source_identifier, day_dir_name, camera, filename, entry, raw_dir
            )
            existing = (
                session.query(Image)
                .filter_by(source_identifier=source_identifier)
                .one_or_none()
            )
            if existing is None:
                session.add(Image(mission_id=mission.id, **values))
                created += 1
            elif any(getattr(existing, k) != v for k, v in values.items()):
                for k, v in values.items():
                    setattr(existing, k, v)
                updated += 1
            else:
                unchanged += 1
        session.commit()  # commit per manifest so an interrupt loses at most one day

    summary = {"created": created, "updated": updated, "unchanged": unchanged}
    logger.info("Persist summary: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Persist day-manifest metadata into PostgreSQL"
    )
    parser.add_argument("--data-root", default=None,
                        help="Override SOHO_DATA_ROOT for this run")
    args = parser.parse_args(argv)

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    raw_dir = (Path(args.data_root) / "raw" / "soho" / "lasco"
               if args.data_root else config.RAW_DIR)
    if not raw_dir.exists():
        logger.error("Raw data directory does not exist: %s", raw_dir)
        return 1

    with SessionLocal() as session:
        persist_manifests(session, raw_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
