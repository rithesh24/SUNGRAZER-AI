"""Evidence builder (tracker §19 subset): one structured, provenance-complete
JSON package per candidate, assembled from the DB index + the file store.

Usage (one-off inspection):
    python -m dataset.evidence --track-id seq36_e91e65277cbe11c1_00104

The FastAPI layer serves ``build_evidence`` directly; nothing is persisted.
Every claim in the package is traceable: track record and Motion DNA are
quoted verbatim from the config-hashed pipeline files, scores carry their
run IDs, and event attribution quotes the Sungrazer source line.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Candidate, ImageSequence, ModelPrediction
from db.session import SessionLocal

logger = logging.getLogger(__name__)


def _sequence_of(track_id: str) -> int:
    match = re.match(r"seq(\d+)_", track_id)
    if not match:
        raise ValueError(f"track id without seq prefix: {track_id}")
    return int(match.group(1))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _event_for(track_id: str, seq_id: int, labels: dict) -> dict | None:
    """Event attribution: positive membership, else day registration."""
    for event in labels.get("events", []):
        for day, rec in event.get("days", {}).items():
            if track_id in rec.get("positive_track_ids", []):
                return {"soho_number": event["soho_number"],
                        "relation": "confirmed_positive", "day": day,
                        "group": event.get("group"),
                        "source_line": event.get("source_line"),
                        "notes": event.get("notes") or None}
    for event in labels.get("events", []):
        for day, rec in event.get("days", {}).items():
            if rec.get("sequence_id") == seq_id and not rec.get("positive_track_ids"):
                return {"soho_number": event["soho_number"],
                        "relation": "unrecovered_event_day", "day": day,
                        "group": event.get("group"),
                        "source_line": event.get("source_line"),
                        "notes": event.get("notes") or None}
    return None


def build_evidence(session: Session, data_root: Path, track_id: str) -> dict:
    """Assemble the full evidence package for one candidate track.

    Raises KeyError if the track is not in the candidates table and
    FileNotFoundError if a pipeline file is missing (never fabricates).
    """
    candidate = session.execute(
        select(Candidate).where(Candidate.track_id == track_id)
    ).scalar_one_or_none()
    if candidate is None:
        raise KeyError(f"candidate {track_id!r} not in database")
    seq_id = candidate.sequence_id
    sequence = session.get(ImageSequence, seq_id)

    processed = data_root / "processed"
    tracks_doc = _load_json(processed / "tracks" / f"seq_{seq_id}" / "tracks.json")
    track = next((t for t in tracks_doc["tracks"] if t["track_id"] == track_id), None)
    if track is None:
        raise FileNotFoundError(f"{track_id} missing from seq_{seq_id} tracks.json")
    dna_doc = _load_json(processed / "motion_dna" / f"seq_{seq_id}" / "motion_dna.json")
    labels = _load_json(data_root / "dataset" / "v1" / "labels.json")

    predictions = session.execute(
        select(ModelPrediction)
        .where(ModelPrediction.candidate_id == candidate.id)
        .order_by(ModelPrediction.run_id)
    ).scalars().all()

    crop_path = (processed / "candidate_crops" / f"seq_{seq_id}"
                 / f"{track_id}.npz")
    return {
        "track_id": track_id,
        "candidate": {
            "sequence_id": seq_id,
            "instrument": sequence.instrument if sequence else None,
            "start_time": candidate.start_time.isoformat() if candidate.start_time else None,
            "end_time": candidate.end_time.isoformat() if candidate.end_time else None,
            "n_frames": candidate.n_frames,
            "label": candidate.label,
            "status": candidate.status,
        },
        "track": track,  # verbatim: positions, timestamps, photometry, residuals
        "motion_dna": dna_doc["features"].get(track_id),
        "predictions": [{"run_id": p.run_id, "model_version": p.model_version,
                         "score": p.score} for p in predictions],
        "event": _event_for(track_id, seq_id, labels),
        "provenance": {
            "tracks_config_hash": tracks_doc.get("config_hash"),
            "motion_config_hash": tracks_doc.get("motion_config_hash"),
            "dna_config_hash": dna_doc.get("config_hash"),
            "sun_center_xy": dna_doc.get("sun_center_xy"),
            "crop_stack": (crop_path.relative_to(data_root).as_posix()
                           if crop_path.exists() else None),
            "label_format": labels.get("label_format"),
        },
        "caveats": [
            "Scores are ranking signals, not detection probabilities; "
            "model calibration on unseen months is weak.",
            "A candidate is not a discovery: confirmation requires human "
            "review and astrometric verification.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump one evidence package")
    parser.add_argument("--data-root",
                        default=os.environ.get("SOHO_DATA_ROOT", "data"))
    parser.add_argument("--track-id", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    with SessionLocal() as session:
        evidence = build_evidence(session, Path(args.data_root), args.track_id)
    json.dump(evidence, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
