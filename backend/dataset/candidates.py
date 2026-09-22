"""Persist pipeline tracks into the ``candidates`` table (tracker §9/§16).

Usage:
    python -m dataset.candidates --sequence-id 36 [--sequence-id 37 ...]
    python -m dataset.candidates --all

The file store (``processed/tracks``, ``processed/motion_dna``) remains the
scientific system of record; this loader builds the queryable DB index the
API serves. Idempotent: upserts by unique ``track_id``.

Label semantics (from ``dataset/v1/labels.json``):
- ``comet``    — track is a confirmed positive of a labeled event
- ``excluded`` — track's day is registered to an event with no recovered
  positives (comet below detection floor) so it must not be treated as a
  clean negative
- NULL         — everything else (unverified background track)

Status: ``KNOWN_COMET`` for comet tracks, else ``UNRESOLVED`` (priority
statuses are assigned by ``ml.score`` once model scores exist).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert

from db.models import Candidate
from db.session import SessionLocal

logger = logging.getLogger(__name__)


def _sequence_of(track_id: str) -> int:
    match = re.match(r"seq(\d+)_", track_id)
    if not match:
        raise ValueError(f"track id without seq prefix: {track_id}")
    return int(match.group(1))


def label_maps(data_root: Path) -> tuple[set[str], set[int]]:
    """Return (comet track IDs, sequence IDs excluded from the negative pool)."""
    labels = json.loads((data_root / "dataset" / "v1" / "labels.json")
                        .read_text(encoding="utf-8"))
    comets: set[str] = set()
    excluded: set[int] = set()
    for event in labels["events"]:
        for rec in event["days"].values():
            if rec["positive_track_ids"]:
                comets.update(rec["positive_track_ids"])
            elif rec["sequence_id"] is not None:
                excluded.add(rec["sequence_id"])
    return comets, excluded


def candidate_rows(data_root: Path, seq_id: int,
                   comets: set[str], excluded: set[int]) -> list[dict]:
    tracks_file = data_root / "processed" / "tracks" / f"seq_{seq_id}" / "tracks.json"
    dna_file = (data_root / "processed" / "motion_dna" / f"seq_{seq_id}"
                / "motion_dna.json")
    tracks = json.loads(tracks_file.read_text(encoding="utf-8"))["tracks"]
    features = json.loads(dna_file.read_text(encoding="utf-8"))["features"]
    rows = []
    for track in tracks:
        track_id = track["track_id"]
        if track_id in comets:
            label, status = "comet", "KNOWN_COMET"
        elif seq_id in excluded:
            label, status = "excluded", "UNRESOLVED"
        else:
            label, status = None, "UNRESOLVED"
        rows.append({
            "track_id": track_id,
            "sequence_id": seq_id,
            "n_frames": track["n_frames"],
            "start_time": datetime.fromisoformat(track["timestamps"][0]),
            "end_time": datetime.fromisoformat(track["timestamps"][-1]),
            "features": features.get(track_id),
            "label": label,
            "status": status,
        })
    return rows


def persist_sequences(data_root: Path, seq_ids: list[int]) -> dict:
    comets, excluded = label_maps(data_root)
    summary = {"sequences": 0, "tracks": 0}
    with SessionLocal() as session:
        for seq_id in seq_ids:
            rows = candidate_rows(data_root, seq_id, comets, excluded)
            stmt = insert(Candidate).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["track_id"],
                set_={c: stmt.excluded[c]
                      for c in ("n_frames", "start_time", "end_time",
                                "features", "label", "status")},
            )
            session.execute(stmt)
            summary["sequences"] += 1
            summary["tracks"] += len(rows)
            logger.info("seq %d: %d candidates upserted", seq_id, len(rows))
        session.commit()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Persist tracks as candidates")
    parser.add_argument("--data-root",
                        default=os.environ.get("SOHO_DATA_ROOT", "data"))
    parser.add_argument("--sequence-id", type=int, action="append", default=[])
    parser.add_argument("--all", action="store_true",
                        help="every sequence with tracks on disk")
    args = parser.parse_args(argv)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    data_root = Path(args.data_root)
    if args.all:
        seq_ids = sorted(int(p.name.removeprefix("seq_"))
                         for p in (data_root / "processed" / "tracks").iterdir()
                         if p.name.startswith("seq_"))
    else:
        seq_ids = args.sequence_id
    if not seq_ids:
        parser.error("give --sequence-id or --all")
    summary = persist_sequences(data_root, seq_ids)
    logger.info("Candidate persist summary: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
