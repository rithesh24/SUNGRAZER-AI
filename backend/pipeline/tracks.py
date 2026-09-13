"""Candidate tracking: per-frame detections -> multi-frame tracks.

Usage:
    python -m pipeline.tracks --sequence-id N [--data-root PATH]

Associates the motion-stage detections (``processed/motion/seq_<id>/
detections.json``) across frames into tracks — the first persistent
candidate representation (techspec sections 12/13). The strategy is the
simplest explainable one (techspec 13.3): greedy nearest-neighbor
assignment of detections to velocity-predicted track positions.

Per frame, in chronological order:

1. Every open track predicts its current position: last position plus its
   mean velocity (px/s over the whole track) times the elapsed time.
   Single-detection tracks predict their only position.
2. All (track, detection) pairs within the gating radius are assigned
   greedily by ascending distance, one-to-one.
3. Unassigned detections open new tracks; tracks unmatched for more than
   ``max_missed_frames`` consecutive frames are terminated (they coast
   through shorter dropouts, so temporary missed detections do not split
   a track).

Cosmic rays terminate as single-detection tracks and are dropped by
``min_track_frames``; drifting stars form long tracks and are KEPT — they
are the future hard-negative catalog, and downstream ranking (not this
recall-first stage) separates them from real movers.

Velocities are px/s because the LASCO cadence is irregular (nominal ~12
min, gaps up to the sequence-split threshold).

Output: ``processed/tracks/seq_<id>/tracks.json`` — config + hash, the
motion-stage config hash (provenance chain), and per-track frames,
timestamps, positions, bboxes, photometry, velocity and residual stats.

Known failure modes: crossing tracks can swap identities (greedy has no
global assignment); dense detection clumps can seed spurious short tracks;
a mover faster than ``max_link_px`` per frame step is never linked.
No split/merge handling (one-to-one only) — revisit if real candidates
demand it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from pipeline.fileio import atomic_write_text
from pipeline.fits_io import parse_date_obs

logger = logging.getLogger(__name__)

TRACKS_VERSION = "track_v1"


@dataclass(frozen=True)
class TracksConfig:
    """Deterministic configuration for one tracking run."""

    version: str = TRACKS_VERSION
    max_link_px: float = 10.0    # gate radius, detection vs predicted position
    max_missed_frames: int = 2   # coast through up to this many frames
    min_track_frames: int = 5    # persistence filter for kept tracks

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class _Track:
    """Mutable in-progress track (internal to build_tracks)."""

    frames: list[str] = field(default_factory=list)
    times: list[datetime] = field(default_factory=list)
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    detections: list[dict] = field(default_factory=list)
    missed: int = 0
    residuals: list[float] = field(default_factory=list)

    def predict(self, at_time: datetime) -> tuple[float, float]:
        """Position at ``at_time`` from the track's mean velocity."""
        if len(self.times) < 2:
            return self.xs[-1], self.ys[-1]
        span = (self.times[-1] - self.times[0]).total_seconds()
        dt = (at_time - self.times[-1]).total_seconds()
        vx = (self.xs[-1] - self.xs[0]) / span
        vy = (self.ys[-1] - self.ys[0]) / span
        return self.xs[-1] + vx * dt, self.ys[-1] + vy * dt

    def add(self, frame: str, time: datetime, det: dict, residual: float) -> None:
        self.frames.append(frame)
        self.times.append(time)
        self.xs.append(det["x"])
        self.ys.append(det["y"])
        self.detections.append(det)
        self.residuals.append(residual)
        self.missed = 0


def build_tracks(frame_names: list[str], detections_by_frame: dict[str, list[dict]],
                 timestamps: dict[str, datetime], config: TracksConfig) -> list[_Track]:
    """Associate detections into tracks. Pure; frames must be chronological.

    Returns every finished track (including single-detection ones); the
    caller applies ``min_track_frames``.
    """
    open_tracks: list[_Track] = []
    finished: list[_Track] = []

    for name in frame_names:
        now = timestamps[name]
        detections = detections_by_frame[name]

        # Distance of every open track's prediction to every detection.
        pairs = []  # (distance, track_idx, det_idx)
        if open_tracks and detections:
            predictions = np.array([t.predict(now) for t in open_tracks])
            positions = np.array([[d["x"], d["y"]] for d in detections])
            dists = np.hypot(
                predictions[:, 0:1] - positions[None, :, 0],
                predictions[:, 1:2] - positions[None, :, 1])
            ti, di = np.nonzero(dists <= config.max_link_px)
            pairs = sorted(zip(dists[ti, di], ti, di))

        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()
        for dist, t_idx, d_idx in pairs:  # greedy one-to-one by distance
            if t_idx in assigned_tracks or d_idx in assigned_dets:
                continue
            open_tracks[t_idx].add(name, now, detections[d_idx], float(dist))
            assigned_tracks.add(t_idx)
            assigned_dets.add(d_idx)

        # Unmatched tracks age; too-old ones terminate.
        still_open = []
        for i, track in enumerate(open_tracks):
            if i in assigned_tracks:
                still_open.append(track)
                continue
            track.missed += 1
            if track.missed > config.max_missed_frames:
                finished.append(track)
            else:
                still_open.append(track)
        open_tracks = still_open

        # Unmatched detections seed new tracks.
        for d_idx, det in enumerate(detections):
            if d_idx not in assigned_dets:
                track = _Track()
                track.add(name, now, det, 0.0)
                open_tracks.append(track)

    finished.extend(open_tracks)
    return finished


def track_to_dict(track: _Track, track_id: str) -> dict:
    """Serializable track record with summary statistics."""
    duration = (track.times[-1] - track.times[0]).total_seconds()
    displacement = float(np.hypot(track.xs[-1] - track.xs[0],
                                  track.ys[-1] - track.ys[0]))
    step_residuals = track.residuals[1:]  # first point has no prediction
    return {
        "track_id": track_id,
        "n_frames": len(track.frames),
        "frames": track.frames,
        "timestamps": [t.isoformat() for t in track.times],
        "positions": [[x, y] for x, y in zip(track.xs, track.ys)],
        "bboxes": [d["bbox"] for d in track.detections],
        "areas": [d["area"] for d in track.detections],
        "peaks": [d["peak"] for d in track.detections],
        "fluxes": [d["flux"] for d in track.detections],
        "snrs": [d["snr"] for d in track.detections],
        "duration_s": round(duration, 3),
        "displacement_px": round(displacement, 2),
        "mean_speed_px_s": round(displacement / duration, 6) if duration > 0 else 0.0,
        "mean_residual_px": round(float(np.mean(step_residuals)), 3) if step_residuals else 0.0,
    }


def load_timestamps(sequence_id: int, data_root: Path,
                    frame_names: list[str]) -> dict[str, datetime]:
    """Observation time per frame, via preprocess manifest -> raw day manifest.

    File-side provenance chain (works without the database): the preprocess
    manifest records each frame's raw source path; the raw day manifest
    beside that file records the FITS DATE-OBS captured at ingestion.
    """
    prep_manifest = (data_root / "processed" / "normalized"
                     / f"seq_{sequence_id}" / "preprocess_manifest.json")
    if not prep_manifest.exists():
        raise FileNotFoundError(f"{prep_manifest} missing — run pipeline.preprocess first")
    prep_frames = json.loads(prep_manifest.read_text(encoding="utf-8"))["frames"]

    day_manifests: dict[Path, dict] = {}
    timestamps: dict[str, datetime] = {}
    for name in frame_names:
        entry = prep_frames.get(name)
        if entry is None:
            raise KeyError(f"Frame {name} not in {prep_manifest}")
        source = Path(entry["source"])
        manifest_path = data_root / source.parent / "manifest.json"
        if manifest_path not in day_manifests:
            day_manifests[manifest_path] = json.loads(
                manifest_path.read_text(encoding="utf-8"))
        raw = day_manifests[manifest_path].get(source.name, {}).get("date_obs")
        parsed = parse_date_obs(raw)
        if parsed is None:
            raise ValueError(f"No parseable date_obs for {source} in {manifest_path}")
        timestamps[name] = parsed
    return timestamps


def tracks_sequence(sequence_id: int, data_root: Path,
                    config: TracksConfig | None = None) -> dict:
    """Track one sequence's detections and save the result."""
    config = config or TracksConfig()
    detections_path = (data_root / "processed" / "motion"
                       / f"seq_{sequence_id}" / "detections.json")
    if not detections_path.exists():
        raise FileNotFoundError(f"{detections_path} missing — run pipeline.motion first")
    motion = json.loads(detections_path.read_text(encoding="utf-8"))
    detections_by_frame = {name: entry["detections"]
                           for name, entry in motion["frames"].items()}
    timestamps = load_timestamps(sequence_id, data_root, list(detections_by_frame))
    frame_names = sorted(detections_by_frame, key=lambda n: timestamps[n])

    all_tracks = build_tracks(frame_names, detections_by_frame, timestamps, config)
    kept = [t for t in all_tracks if len(t.frames) >= config.min_track_frames]
    kept.sort(key=lambda t: (t.times[0], t.xs[0], t.ys[0]))
    # Stable within (sequence, config): deterministic input order + sort.
    records = [track_to_dict(t, f"seq{sequence_id}_{config.config_hash()}_{i:05d}")
               for i, t in enumerate(kept)]

    result = {
        "sequence_id": sequence_id,
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "motion_config_hash": motion["config_hash"],
        "n_frames": len(frame_names),
        "n_detections": sum(len(d) for d in detections_by_frame.values()),
        "n_tracks_raw": len(all_tracks),
        "n_tracks_kept": len(records),
        "tracks": records,
    }
    out_path = data_root / "processed" / "tracks" / f"seq_{sequence_id}" / "tracks.json"
    atomic_write_text(out_path, json.dumps(result, indent=2, sort_keys=True))

    summary = {k: result[k] for k in
               ("sequence_id", "n_frames", "n_detections", "n_tracks_raw", "n_tracks_kept")}
    logger.info("Tracking summary: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Associate motion detections into candidate tracks")
    parser.add_argument("--sequence-id", type=int, required=True)
    parser.add_argument("--data-root", default=os.environ.get("SOHO_DATA_ROOT", "data"))
    parser.add_argument("--min-track-frames", type=int,
                        default=TracksConfig.min_track_frames)
    args = parser.parse_args(argv)

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    tracks_sequence(args.sequence_id, Path(args.data_root),
                    TracksConfig(min_track_frames=args.min_track_frames))
    return 0


if __name__ == "__main__":
    sys.exit(main())
