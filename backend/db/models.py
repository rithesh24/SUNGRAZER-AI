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
