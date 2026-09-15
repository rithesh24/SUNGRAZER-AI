"""Candidate crop extraction: tracks -> per-track temporal crop stacks.

Usage:
    python -m ml.crops --sequence-id N [--data-root PATH]

Model input format (``crops_v1``, the tracker-section-14 "model input
format" decision):

For every kept track of a sequence, a float32 stack ``[T, K, K]`` of
K x K crops (default 32) cut from the **median-background-subtracted**
registered frames, centered on the track's per-frame detection position.
The background is the same self-excluding temporal median the detection
stage uses (``pipeline.motion.temporal_background``, same default window),
so the model sees exactly the residual signal detections were made from —
static corona and star field removed, moving sources preserved.

- Crops exist only for frames where the track has a detection (coasted
  frames have no position to crop at); per-frame timestamps are stored so
  the temporal model can see the true cadence.
- Pixels outside the frame or outside the registered ``valid_mask`` are 0
  (consistent with the preprocessing masks); ``valid_fractions`` records
  how much of each crop is real data.
- Values stay in raw residual DN/s — normalization is a training-time
  choice (ml.dataset), not baked into the stored artifact.

Output (techspec section 31 layout):

    data/processed/candidate_crops/seq_<id>/<track_id>.npz
        crops            float32 [T, K, K]
        positions        float32 [T, 2]   (x, y in registered-frame pixels)
        frames           str    [T]      (8-digit frame ids)
        timestamps       str    [T]      (ISO 8601 UTC)
        valid_fractions  float32 [T]
    data/processed/candidate_crops/seq_<id>/crops_manifest.json
        config + hash, source tracks config hash, per-track file list

Re-running with an unchanged config and complete outputs is a no-op
(manifest hash + file existence check).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pipeline.fileio import atomic_write_bytes_via, atomic_write_text
from pipeline.motion import load_registered_sequence, temporal_background

logger = logging.getLogger(__name__)

CROPS_VERSION = "crops_v1"


@dataclass(frozen=True)
class CropConfig:
    """Deterministic configuration for one crop-extraction run."""

    version: str = CROPS_VERSION
    crop_size_px: int = 32          # K: crop side length, even
    background_method: str = "median"
    background_window: int = 5      # matches MotionConfig.window default

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cut_crop(residual: np.ndarray, valid_mask: np.ndarray,
             x: float, y: float, size: int) -> tuple[np.ndarray, float]:
    """Cut a ``size`` x ``size`` crop centered on (x, y), zero-padded at edges.

    Returns (crop, valid_fraction). Invalid-mask pixels are zeroed.
    """
    half = size // 2
    cx, cy = int(round(x)), int(round(y))
    crop = np.zeros((size, size), dtype=np.float32)
    y0, y1 = cy - half, cy - half + size
    x0, x1 = cx - half, cx - half + size
    sy0, sy1 = max(0, y0), min(residual.shape[0], y1)
    sx0, sx1 = max(0, x0), min(residual.shape[1], x1)
    if sy0 >= sy1 or sx0 >= sx1:
        return crop, 0.0
    region = residual[sy0:sy1, sx0:sx1] * valid_mask[sy0:sy1, sx0:sx1]
    crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = region
    valid_fraction = float(valid_mask[sy0:sy1, sx0:sx1].sum()) / (size * size)
    return crop, valid_fraction


def extract_sequence_crops(sequence_id: int, data_root: Path,
                           config: CropConfig) -> dict:
    """Extract crop stacks for every kept track of a sequence."""
    tracks_file = (data_root / "processed" / "tracks"
                   / f"seq_{sequence_id}" / "tracks.json")
    if not tracks_file.exists():
        raise FileNotFoundError(f"{tracks_file} missing — run pipeline.tracks first")
    tracks_doc = json.loads(tracks_file.read_text(encoding="utf-8"))

    out_dir = data_root / "processed" / "candidate_crops" / f"seq_{sequence_id}"
    manifest_path = out_dir / "crops_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("config_hash") == config.config_hash()
                and manifest.get("tracks_config_hash") == tracks_doc["config_hash"]
                and all((out_dir / f).exists() for f in manifest["track_files"])):
            logger.info("seq %d: crops up to date (%d tracks), skipping",
                        sequence_id, len(manifest["track_files"]))
            return manifest

    registered_dir = data_root / "processed" / "registered" / f"seq_{sequence_id}"
    names, frames, masks = load_registered_sequence(registered_dir)
    index_of = {name: i for i, name in enumerate(names)}

    # Residuals are computed lazily per frame index and cached, since most
    # frames host detections from many tracks.
    residual_cache: dict[int, np.ndarray] = {}

    def residual(i: int) -> np.ndarray:
        if i not in residual_cache:
            background = temporal_background(frames, i, config.background_window)
            residual_cache[i] = frames[i] - background
        return residual_cache[i]

    track_files = []
    for track in tracks_doc["tracks"]:
        kept: list[tuple[np.ndarray, float, list[float], str, str]] = []
        for frame_id, position, timestamp in zip(
                track["frames"], track["positions"], track["timestamps"]):
            i = index_of.get(str(frame_id))
            if i is None:  # frame not in the registered set
                continue
            crop, fraction = cut_crop(residual(i), masks[i],
                                      position[0], position[1],
                                      config.crop_size_px)
            kept.append((crop, fraction, position, str(frame_id), timestamp))
        if not kept:
            logger.warning("track %s: no registered frames, skipped",
                           track["track_id"])
            continue
        out_path = out_dir / f"{track['track_id']}.npz"
        payload = {
            "crops": np.stack([k[0] for k in kept]),
            "positions": np.asarray([k[2] for k in kept], dtype=np.float32),
            "frames": np.asarray([k[3] for k in kept]),
            "timestamps": np.asarray([k[4] for k in kept]),
            "valid_fractions": np.asarray([k[1] for k in kept],
                                          dtype=np.float32),
        }
        atomic_write_bytes_via(
            out_path, lambda fh, p=payload: np.savez_compressed(fh, **p))
        track_files.append(out_path.name)

    manifest = {
        "sequence_id": sequence_id,
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "tracks_config_hash": tracks_doc["config_hash"],
        "n_tracks": len(track_files),
        "track_files": sorted(track_files),
    }
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
    logger.info("seq %d: wrote %d crop stacks to %s",
                sequence_id, len(track_files), out_dir)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Candidate crop extraction")
    parser.add_argument("--sequence-id", type=int, required=True)
    parser.add_argument("--data-root",
                        default=os.environ.get("SOHO_DATA_ROOT", "data"))
    parser.add_argument("--crop-size", type=int,
                        default=CropConfig.crop_size_px)
    args = parser.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
    config = CropConfig(crop_size_px=args.crop_size)
    extract_sequence_crops(args.sequence_id, Path(args.data_root), config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
