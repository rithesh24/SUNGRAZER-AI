"""Group ingested images into chronological image sequences.

Usage:
    python -m ingestion.sequences [--gap-minutes N] [--check-only]

Assigns every ungrouped, successfully ingested image (``images.sequence_id
IS NULL``, ``ingestion_status = 'complete'``, ``observation_time`` present)
to an ``image_sequences`` row. Grouping is per instrument ("LASCO/C2" and
"LASCO/C3" never mix) and splits whenever the time between consecutive
frames exceeds the gap threshold.

Idempotent and incremental: already-grouped images are never touched, and a
new image joins an existing sequence when its timestamp falls within the gap
threshold of that sequence's time window — so late backfills extend
sequences instead of duplicating them. Existing sequence IDs are stable
(provenance, claude.md section 32).

Known limitation (ponytail: acceptable until live mode): an image that
bridges two existing sequences extends one of them but does not merge the
two. Merging needs downstream-reference handling and lands with live mode.

Every run finishes with integrity checks over all sequences; failures are
reported and reflected in the exit code.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Image, ImageSequence
from db.session import SessionLocal
from ingestion import config

logger = logging.getLogger(__name__)


def group_by_gap(times: list[datetime], gap: timedelta) -> list[list[int]]:
    """Split chronologically sorted timestamps into runs with gaps <= ``gap``.

    Returns lists of indices into ``times``. Pure function, unit-tested
    offline; the DB code below reuses its semantics for incremental attach.
    """
    if not times:
        return []
    groups: list[list[int]] = [[0]]
    for i in range(1, len(times)):
        if times[i] - times[i - 1] > gap:
            groups.append([i])
        else:
            groups[-1].append(i)
    return groups


def _find_adjacent_sequence(session: Session, instrument: str,
                            t: datetime, gap: timedelta) -> ImageSequence | None:
    """Return an existing sequence whose window [start-gap, end+gap] covers t."""
    return session.scalars(
        select(ImageSequence)
        .where(
            ImageSequence.instrument == instrument,
            ImageSequence.start_time - gap <= t,
            ImageSequence.end_time + gap >= t,
        )
        .order_by(ImageSequence.start_time)
        .limit(1)
    ).first()


def _attach(image: Image, sequence: ImageSequence) -> None:
    image.sequence_id = sequence.id
    sequence.frame_count += 1
    if sequence.start_time is None or image.observation_time < sequence.start_time:
        sequence.start_time = image.observation_time
    if sequence.end_time is None or image.observation_time > sequence.end_time:
        sequence.end_time = image.observation_time


def build_sequences(session: Session, gap: timedelta) -> dict:
    """Assign all ungrouped complete images to sequences. Returns a summary."""
    ungrouped = session.scalars(
        select(Image)
        .where(
            Image.sequence_id.is_(None),
            Image.ingestion_status == "complete",
            Image.observation_time.is_not(None),
        )
        .order_by(Image.instrument, Image.observation_time, Image.source_identifier)
    ).all()

    skipped = session.scalars(
        select(Image.id).where(
            Image.sequence_id.is_(None),
            (Image.ingestion_status != "complete") | (Image.observation_time.is_(None)),
        )
    ).all()
    if skipped:
        logger.warning("%d ungrouped images skipped (failed ingestion or no "
                       "observation_time)", len(skipped))

    created = attached = 0
    current: ImageSequence | None = None
    for image in ungrouped:
        # Reuse the in-progress sequence when this frame continues it
        # (same instrument, within gap of its end — images arrive sorted).
        if (current is not None and current.instrument == image.instrument
                and image.observation_time - current.end_time <= gap):
            _attach(image, current)
            attached += 1
            continue
        current = _find_adjacent_sequence(session, image.instrument,
                                          image.observation_time, gap)
        if current is None:
            current = ImageSequence(
                mission_id=image.mission_id,
                instrument=image.instrument,
                start_time=image.observation_time,
                end_time=image.observation_time,
                frame_count=0,
                status="PENDING",
            )
            session.add(current)
            session.flush()  # need current.id for _attach
            created += 1
        _attach(image, current)
        attached += 1

    session.commit()
    summary = {"grouped": attached, "sequences_created": created,
               "skipped": len(skipped)}
    logger.info("Sequence grouping summary: %s", summary)
    return summary


def check_sequences(session: Session, gap: timedelta) -> list[str]:
    """Integrity checks over every sequence. Returns problem descriptions."""
    problems: list[str] = []
    for seq in session.scalars(select(ImageSequence)).all():
        frames = session.scalars(
            select(Image)
            .where(Image.sequence_id == seq.id)
            .order_by(Image.observation_time, Image.source_identifier)
        ).all()
        if len(frames) != seq.frame_count:
            problems.append(f"sequence {seq.id}: frame_count={seq.frame_count} "
                            f"but {len(frames)} member images")
        if not frames:
            continue
        if frames[0].observation_time != seq.start_time:
            problems.append(f"sequence {seq.id}: start_time mismatch")
        if frames[-1].observation_time != seq.end_time:
            problems.append(f"sequence {seq.id}: end_time mismatch")
        for f in frames:
            if f.instrument != seq.instrument:
                problems.append(f"sequence {seq.id}: image {f.id} instrument "
                                f"{f.instrument} != {seq.instrument}")
            if f.observation_time is None:
                problems.append(f"sequence {seq.id}: image {f.id} has no "
                                f"observation_time")
        for a, b in zip(frames, frames[1:]):
            if a.observation_time and b.observation_time \
                    and b.observation_time - a.observation_time > gap:
                problems.append(
                    f"sequence {seq.id}: internal gap "
                    f"{b.observation_time - a.observation_time} between "
                    f"images {a.id} and {b.id} exceeds {gap}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Group ingested images into chronological sequences")
    parser.add_argument("--gap-minutes", type=float,
                        default=config.SEQUENCE_GAP_MINUTES,
                        help="Maximum minutes between consecutive frames")
    parser.add_argument("--check-only", action="store_true",
                        help="Run integrity checks without grouping")
    args = parser.parse_args(argv)
    if args.gap_minutes <= 0:
        parser.error("--gap-minutes must be positive")

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    gap = timedelta(minutes=args.gap_minutes)
    with SessionLocal() as session:
        if not args.check_only:
            build_sequences(session, gap)
        problems = check_sequences(session, gap)
        for problem in problems:
            logger.error("Integrity: %s", problem)
        if problems:
            return 1
        logger.info("Integrity checks passed for all sequences")
        return 0


if __name__ == "__main__":
    sys.exit(main())
