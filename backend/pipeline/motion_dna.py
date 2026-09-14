"""Motion DNA: behavioral features per candidate track (techspec section 14).

Usage:
    python -m pipeline.motion_dna --sequence-id N [--data-root PATH]

Reads ``processed/tracks/seq_<id>/tracks.json`` and computes, per track, a
flat feature record — the deterministic behavioral representation consumed
by candidate ranking. Output: ``processed/motion_dna/seq_<id>/
motion_dna.json`` (config + hash, tracks-stage config hash, features keyed
by track_id). PostgreSQL persistence (``motion_features``) lands with the
candidates tables; until then this file is the feature store.

Feature reference (units; all positions in registered-frame pixels, times
in seconds; image coordinates: x right, y DOWN, so direction angles are
clockwise-positive on screen):

- ``n_frames``            detections in the track (persistence).
- ``frame_coverage``      n_frames / sequence frames between first and last
                          detection inclusive; 1.0 = no missed frames.
- ``duration_s``          last minus first detection time.
- ``path_length_px``      sum of step distances.
- ``displacement_px``     straight-line first-to-last distance.
- ``speed_mean_px_s`` / ``speed_std_px_s``
                          mean/std of instantaneous step speeds.
- ``direction_deg``       overall atan2(dy, dx) of the net displacement,
                          degrees in [-180, 180]; null when displacement 0.
- ``direction_consistency``
                          mean resultant length of unit step vectors in
                          [0, 1]; 1 = perfectly straight, ~0 = random walk.
- ``accel_mean_px_s2``    mean |Δvelocity| / Δt between consecutive steps.
- ``curvature_rad_px``    total absolute heading change / path length;
                          0 for a straight track.
- ``linear_rms_px``       RMS residual of a constant-velocity least-squares
                          fit of x(t) and y(t) — trajectory smoothness;
                          near 0 = consistent linear motion.
- ``flux_mean`` / ``flux_cv``
                          mean and coefficient of variation (std/mean) of
                          per-frame integrated flux (brightness stability).
- ``flux_slope_frac_h``   linear flux trend as fraction of mean flux per
                          hour (+ = brightening).
- ``area_mean_px`` / ``area_slope_frac_h``
                          same for detection area (apparent size).
- ``r_min_px`` / ``r_mean_px`` / ``r_max_px``
                          distance from the reference-frame sun center.
- ``radial_speed_px_s``   (r_last − r_first) / duration; negative = inbound
                          (sunward — the sungrazer signature).

Radial features are null when the reference frame has no sun center
(explicit uncertainty, never a guessed center). Features needing ≥3 points
(acceleration, curvature) are null on shorter tracks; ranges are checked by
``validate_features`` and violations fail the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from pipeline.fileio import atomic_write_text
from pipeline.fits_io import load_science_image

logger = logging.getLogger(__name__)

DNA_VERSION = "dna_v1"


@dataclass(frozen=True)
class MotionDnaConfig:
    """Deterministic configuration for one Motion DNA run."""

    version: str = DNA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _linear_slope(t: np.ndarray, values: np.ndarray) -> float:
    """Least-squares slope of values vs t (t in seconds)."""
    return float(np.polyfit(t, values, 1)[0])


def compute_features(track: dict, sequence_frames: list[str],
                     sun_center: tuple[float, float] | None) -> dict:
    """Motion DNA for one track record from tracks.json. Pure function.

    ``sequence_frames`` is the chronological frame-name list of the whole
    sequence (for frame coverage); ``sun_center`` is 0-based pixel coords
    of the registered reference frame, or None if unknown.
    """
    times = [datetime.fromisoformat(ts) for ts in track["timestamps"]]
    t = np.array([(ts - times[0]).total_seconds() for ts in times])
    xy = np.asarray(track["positions"], dtype=float)
    flux = np.asarray(track["fluxes"], dtype=float)
    area = np.asarray(track["areas"], dtype=float)
    n = len(t)

    steps = np.diff(xy, axis=0)
    dts = np.diff(t)
    step_len = np.hypot(steps[:, 0], steps[:, 1])
    path_length = float(step_len.sum())
    displacement = float(np.hypot(*(xy[-1] - xy[0])))
    duration = float(t[-1] - t[0])

    first_idx = sequence_frames.index(track["frames"][0])
    last_idx = sequence_frames.index(track["frames"][-1])
    coverage = n / (last_idx - first_idx + 1)

    velocities = steps / dts[:, None]
    speeds = step_len / dts
    moving = step_len > 0

    features: dict = {
        "n_frames": n,
        "frame_coverage": round(coverage, 4),
        "duration_s": round(duration, 3),
        "path_length_px": round(path_length, 2),
        "displacement_px": round(displacement, 2),
        "speed_mean_px_s": round(float(speeds.mean()), 6),
        "speed_std_px_s": round(float(speeds.std()), 6),
        "direction_deg": (round(math.degrees(math.atan2(*(xy[-1] - xy[0])[::-1])), 2)
                          if displacement > 0 else None),
        "direction_consistency": None,
        "accel_mean_px_s2": None,
        "curvature_rad_px": None,
        "linear_rms_px": None,
        "flux_mean": round(float(flux.mean()), 4),
        "flux_cv": round(float(flux.std() / flux.mean()), 4) if flux.mean() > 0 else None,
        "flux_slope_frac_h": (round(_linear_slope(t, flux) * 3600 / flux.mean(), 5)
                              if duration > 0 and flux.mean() > 0 else None),
        "area_mean_px": round(float(area.mean()), 2),
        "area_slope_frac_h": (round(_linear_slope(t, area) * 3600 / area.mean(), 5)
                              if duration > 0 and area.mean() > 0 else None),
    }

    if moving.any():
        units = steps[moving] / step_len[moving, None]
        resultant = np.hypot(*units.mean(axis=0))
        features["direction_consistency"] = round(float(resultant), 4)

    if n >= 3:
        dv = np.diff(velocities, axis=0)
        mid_dt = (dts[:-1] + dts[1:]) / 2
        features["accel_mean_px_s2"] = round(
            float((np.hypot(dv[:, 0], dv[:, 1]) / mid_dt).mean()), 8)
        headings = np.arctan2(steps[moving][:, 1], steps[moving][:, 0])
        if len(headings) >= 2 and path_length > 0:
            turns = np.abs(np.angle(np.exp(1j * np.diff(headings))))
            features["curvature_rad_px"] = round(float(turns.sum() / path_length), 6)
        # Constant-velocity fit residual over both axes.
        residuals = []
        for axis in (0, 1):
            fit = np.polyval(np.polyfit(t, xy[:, axis], 1), t)
            residuals.append(xy[:, axis] - fit)
        features["linear_rms_px"] = round(
            float(np.sqrt(np.mean(np.concatenate(residuals) ** 2))), 4)

    if sun_center is not None:
        r = np.hypot(xy[:, 0] - sun_center[0], xy[:, 1] - sun_center[1])
        features.update({
            "r_min_px": round(float(r.min()), 2),
            "r_mean_px": round(float(r.mean()), 2),
            "r_max_px": round(float(r.max()), 2),
            "radial_speed_px_s": (round(float((r[-1] - r[0]) / duration), 6)
                                  if duration > 0 else None),
        })
    else:
        features.update({"r_min_px": None, "r_mean_px": None,
                         "r_max_px": None, "radial_speed_px_s": None})
    return features


# (feature, min, max) — null is always allowed; violations fail the run.
_RANGES = [
    ("frame_coverage", 0.0, 1.0),
    ("direction_consistency", 0.0, 1.0),
    ("direction_deg", -180.0, 180.0),
    ("duration_s", 0.0, math.inf),
    ("path_length_px", 0.0, math.inf),
    ("displacement_px", 0.0, math.inf),
    ("speed_mean_px_s", 0.0, math.inf),
    ("speed_std_px_s", 0.0, math.inf),
    ("accel_mean_px_s2", 0.0, math.inf),
    ("curvature_rad_px", 0.0, math.inf),
    ("linear_rms_px", 0.0, math.inf),
    ("flux_cv", 0.0, math.inf),
    ("area_mean_px", 0.0, math.inf),
    ("r_min_px", 0.0, math.inf),
]


def validate_features(features: dict) -> list[str]:
    """Return a list of range/finiteness violations (empty = valid)."""
    problems = []
    for key, value in features.items():
        if value is None:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            problems.append(f"{key} not finite: {value}")
    for key, lo, hi in _RANGES:
        value = features.get(key)
        if value is not None and not lo <= value <= hi:
            problems.append(f"{key}={value} outside [{lo}, {hi}]")
    if features.get("displacement_px") is not None \
            and features.get("path_length_px") is not None \
            and features["displacement_px"] > features["path_length_px"] + 1e-6:
        problems.append("displacement exceeds path length")
    return problems


def reference_sun_center(sequence_id: int, data_root: Path) -> tuple[float, float] | None:
    """0-based sun center of the sequence's registration reference frame.

    Registered positions live in the reference frame's coordinates, so its
    header CRPIX (1-based) minus 1 is the right center for radial features.
    Returns None (features degrade to null) when the header lacks one.
    """
    reg_manifest = (data_root / "processed" / "registered"
                    / f"seq_{sequence_id}" / "registration_manifest.json")
    reference = json.loads(reg_manifest.read_text(encoding="utf-8"))["reference"]
    prep_manifest = (data_root / "processed" / "normalized"
                     / f"seq_{sequence_id}" / "preprocess_manifest.json")
    source = json.loads(prep_manifest.read_text(encoding="utf-8"))["frames"][reference]["source"]
    image = load_science_image(data_root / source)
    if image.sun_center is None:
        logger.warning("Reference frame %s has no sun center; radial features null",
                       reference)
        return None
    return image.sun_center[0] - 1.0, image.sun_center[1] - 1.0


def motion_dna_sequence(sequence_id: int, data_root: Path,
                        config: MotionDnaConfig | None = None) -> dict:
    """Compute and save Motion DNA for every track of one sequence."""
    config = config or MotionDnaConfig()
    tracks_path = (data_root / "processed" / "tracks"
                   / f"seq_{sequence_id}" / "tracks.json")
    if not tracks_path.exists():
        raise FileNotFoundError(f"{tracks_path} missing — run pipeline.tracks first")
    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))

    reg_manifest = (data_root / "processed" / "registered"
                    / f"seq_{sequence_id}" / "registration_manifest.json")
    reg = json.loads(reg_manifest.read_text(encoding="utf-8"))
    sequence_frames = sorted(name for name, e in reg["frames"].items()
                             if e["status"] in ("ok", "reference"))
    sun_center = reference_sun_center(sequence_id, data_root)

    features_by_track = {}
    invalid = 0
    for track in tracks["tracks"]:
        features = compute_features(track, sequence_frames, sun_center)
        problems = validate_features(features)
        if problems:
            invalid += 1
            logger.error("Track %s invalid features: %s",
                         track["track_id"], problems)
        features_by_track[track["track_id"]] = features

    result = {
        "sequence_id": sequence_id,
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "tracks_config_hash": tracks["config_hash"],
        "sun_center_xy": list(sun_center) if sun_center else None,
        "n_tracks": len(features_by_track),
        "n_invalid": invalid,
        "features": features_by_track,
    }
    out_path = (data_root / "processed" / "motion_dna"
                / f"seq_{sequence_id}" / "motion_dna.json")
    atomic_write_text(out_path, json.dumps(result, indent=2, sort_keys=True))

    summary = {"sequence_id": sequence_id, "tracks": len(features_by_track),
               "invalid": invalid}
    logger.info("Motion DNA summary: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute Motion DNA features for a sequence's tracks")
    parser.add_argument("--sequence-id", type=int, required=True)
    parser.add_argument("--data-root", default=os.environ.get("SOHO_DATA_ROOT", "data"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    summary = motion_dna_sequence(args.sequence_id, Path(args.data_root))
    return 1 if summary["invalid"] else 0


if __name__ == "__main__":
    sys.exit(main())
