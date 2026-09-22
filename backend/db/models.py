"""SQLAlchemy ORM models for the Phase 1 data-foundation tables.

Implements the ``missions``, ``image_sequences``, and ``images`` tables from
techspec section 25. Remaining tables (candidates, trajectories, ...) are
added in later phases via Alembic migrations.

Deviations from the techspec's suggested fields, with reasons:

- ``images.mission_id`` was added (not in the suggested list) because
  ``sequence_id`` is nullable until sequence grouping is implemented, and
  provenance to the mission must never be lost (claude.md section 32).
- ``images.size_bytes`` and ``images.error_message`` were added so the day
  manifests can be backfilled without losing information.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Ingestion statuses mirrored from ingestion.manifest.
INGESTION_STATUSES = ("complete", "failed")

# Sequence lifecycle; only PENDING is used until sequence grouping lands.
SEQUENCE_STATUSES = ("PENDING", "READY", "PROCESSING", "PROCESSED", "FAILED")


class Base(DeclarativeBase):
    """Declarative base shared by all SUNGRAZER AI models."""


class Mission(Base):
    """Source mission/instrument, e.g. SOHO/LASCO."""

    __tablename__ = "missions"
    __table_args__ = (
        UniqueConstraint("mission_name", "instrument_name", name="uq_mission_instrument"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mission_name: Mapped[str] = mapped_column(String(100))
    instrument_name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    sequences: Mapped[list["ImageSequence"]] = relationship(back_populates="mission")
    images: Mapped[list["Image"]] = relationship(back_populates="mission")


class ImageSequence(Base):
    """A group of temporally related images processed together."""

    __tablename__ = "image_sequences"
    __table_args__ = (
        CheckConstraint(
            f"status IN {SEQUENCE_STATUSES}", name="ck_sequence_status"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mission_id: Mapped[int] = mapped_column(ForeignKey("missions.id"), index=True)
    # Same granularity as images.instrument ("LASCO/C3"); a sequence must
    # never mix cameras, so the camera is a property of the sequence itself.
    instrument: Mapped[str] = mapped_column(String(50), index=True)
    start_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    frame_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="PENDING")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    mission: Mapped[Mission] = relationship(back_populates="sequences")
    images: Mapped[list["Image"]] = relationship(back_populates="sequence")


class Image(Base):
    """One ingested source image (metadata only; pixels stay on disk)."""

    __tablename__ = "images"
    __table_args__ = (
        CheckConstraint(
            f"ingestion_status IN {INGESTION_STATUSES}", name="ck_image_ingestion_status"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mission_id: Mapped[int] = mapped_column(ForeignKey("missions.id"), index=True)
    # Null until sequence grouping (tracker section 4) assigns this image.
    sequence_id: Mapped[int | None] = mapped_column(
        ForeignKey("image_sequences.id"), index=True
    )
    # Stable archive identity: "<YYMMDD>/<camera>/<filename>", e.g. "240101/c3/32305434.fts".
    source_identifier: Mapped[str] = mapped_column(String(255), unique=True)
    source_url: Mapped[str] = mapped_column(Text)
    # Relative to SOHO_DATA_ROOT so the data tree can move between machines.
    local_path: Mapped[str] = mapped_column(Text)
    observation_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    # e.g. "LASCO/C3" — camera-level granularity within the mission.
    instrument: Mapped[str] = mapped_column(String(50))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    checksum: Mapped[str | None] = mapped_column(String(64))  # SHA-256 hex
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    ingestion_status: Mapped[str] = mapped_column(String(20))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    mission: Mapped[Mission] = relationship(back_populates="images")
    sequence: Mapped[ImageSequence | None] = relationship(back_populates="images")


# Candidate review statuses (claude.md section 12.3). Only a subset is
# assigned automatically; the rest exist for the (descoped) review workflow.
CANDIDATE_STATUSES = ("HIGH_PRIORITY", "MEDIUM_PRIORITY", "LOW_PRIORITY",
                      "KNOWN_COMET", "LIKELY_ARTIFACT", "UNRESOLVED")

# Human review verdicts — deliberately separate from ``status`` so machine
# ranking and human judgment never overwrite each other.
REVIEW_VERDICTS = ("approved", "rejected", "artifact", "known_object",
                   "uncertain")


class Candidate(Base):
    """One persistent moving-object track from the CV pipeline.

    Mirrors ``processed/tracks/seq_<id>/tracks.json``; the file store stays
    the scientific system of record (full per-frame arrays + provenance
    hashes), this table is the queryable index the API serves. ``features``
    holds the Motion DNA record (dna_v1) verbatim.
    """

    __tablename__ = "candidates"
    __table_args__ = (
        CheckConstraint(f"status IN {CANDIDATE_STATUSES}",
                        name="ck_candidate_status"),
        CheckConstraint(f"review IS NULL OR review IN {REVIEW_VERDICTS}",
                        name="ck_candidate_review"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # e.g. "seq36_e91e65277cbe11c1_00104" — stable, config-hashed (§9).
    track_id: Mapped[str] = mapped_column(String(64), unique=True)
    sequence_id: Mapped[int] = mapped_column(
        ForeignKey("image_sequences.id"), index=True
    )
    n_frames: Mapped[int] = mapped_column(Integer)
    start_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Motion DNA (dna_v1) feature dict, verbatim from motion_dna.json.
    features: Mapped[dict | None] = mapped_column(JSONB)
    # "comet" for confirmed positive tracks (labels.json), "excluded" for
    # tracks whose day is excluded from the negative pool, else null.
    label: Mapped[str | None] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="UNRESOLVED")
    # Human review (S7): verdict + notes, never touched by the pipeline.
    review: Mapped[str | None] = mapped_column(String(20))
    reviewer_notes: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    sequence: Mapped[ImageSequence] = relationship()
    predictions: Mapped[list["ModelPrediction"]] = relationship(
        back_populates="candidate"
    )


class ModelPrediction(Base):
    """One model's (or fusion's) score for one candidate."""

    __tablename__ = "model_predictions"
    __table_args__ = (
        UniqueConstraint("candidate_id", "run_id", name="uq_prediction_run"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(
        ForeignKey("candidates.id"), index=True
    )
    model_version: Mapped[str] = mapped_column(String(50))
    # Run-dir name (e.g. "run_f7ded503..._seed0") or a fusion id.
    run_id: Mapped[str] = mapped_column(String(100))
    score: Mapped[float] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    candidate: Mapped[Candidate] = relationship(back_populates="predictions")
