"""Temporal motion extraction: registered frames -> per-frame detections.

Usage:
    python -m pipeline.motion --sequence-id N [--data-root PATH]
    python -m pipeline.motion --sequence-id N --evaluate

For every registered frame the stage builds a temporal background, subtracts
it, thresholds the residual against a robust per-frame noise estimate, and
extracts connected components as raw detections. This stage optimizes for
RECALL (techspec section 12.2): cosmic rays and star residuals are expected
in the output and are removed later by tracking persistence and ranking.

Background methods (both implemented; see --evaluate):

- ``median``    — pixelwise temporal median of the neighboring frames
  (excluding the frame itself, so a slow-moving object does not subtract
  itself). Removes the static corona and most of the star field.
- ``prev_diff`` — absolute difference with the previous frame. Cheapest,
  but every moving object appears twice (positive + negative lobe) and
  noise is doubled; kept as the comparison baseline.

Detections are stored per sequence in ``processed/motion/seq_<id>/
detections.json`` with the full configuration and per-frame noise stats.

``--evaluate`` injects a synthetic faint moving source into the real
registered frames and reports per-method recovery — the evidence for the
baseline choice (recorded in docs/progress.md, session 2026-09-13).
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

import cv2
import numpy as np

from pipeline.fileio import atomic_write_text

logger = logging.getLogger(__name__)

MOTION_VERSION = "motion_v1"


@dataclass(frozen=True)
class MotionConfig:
    """Deterministic configuration for one motion-extraction run."""

    version: str = MOTION_VERSION
    method: str = "median"          # median | prev_diff
    window: int = 5                 # temporal window (median method), odd
    threshold_sigma: float = 5.0    # detection threshold in robust sigmas
    min_area_px: int = 2            # reject single-pixel hits (cosmic rays)
    max_area_px: int = 2000         # reject huge blobs (corona residuals)
    opening_px: int = 0             # optional morphological opening, 0 = off
    max_detections_per_frame: int = 500   # sanity cap, brightest kept

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def robust_sigma(values: np.ndarray) -> float:
    """Noise scale via median absolute deviation (Gaussian-consistent)."""
    if values.size == 0:
        return 0.0
    med = np.median(values)
    return float(1.4826 * np.median(np.abs(values - med)))


def temporal_background(frames: list[np.ndarray], index: int, window: int) -> np.ndarray:
    """Pixelwise median of up to ``window-1`` neighbors of ``frames[index]``.

    The frame itself is excluded so slow movers are not self-subtracted.
    At sequence edges the window is truncated, never wrapped.
    """
    half = window // 2
    lo, hi = max(0, index - half), min(len(frames), index + half + 1)
    neighbors = [frames[i] for i in range(lo, hi) if i != index]
    return np.median(np.stack(neighbors), axis=0)


def extract_detections(residual: np.ndarray, valid_mask: np.ndarray,
                       config: MotionConfig) -> tuple[list[dict], float]:
    """Threshold a background-subtracted residual and extract components.

    Returns (detections, sigma). Detection: x/y centroid, bbox, area,
    peak residual, integrated flux, and SNR (peak / sigma).
    """
    sigma = robust_sigma(residual[valid_mask])
    if sigma <= 0:
        return [], 0.0
    binary = ((residual > config.threshold_sigma * sigma) & valid_mask).astype(np.uint8)
    if config.opening_px > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.opening_px, config.opening_px))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    detections = []
    for label in range(1, count):  # 0 = background
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not config.min_area_px <= area <= config.max_area_px:
            continue
        component = labels == label
        peak = float(residual[component].max())
        detections.append({
            "x": round(float(centroids[label, 0]), 2),
            "y": round(float(centroids[label, 1]), 2),
            "bbox": [int(stats[label, cv2.CC_STAT_LEFT]),
                     int(stats[label, cv2.CC_STAT_TOP]),
                     int(stats[label, cv2.CC_STAT_WIDTH]),
                     int(stats[label, cv2.CC_STAT_HEIGHT])],
            "area": area,
            "peak": round(peak, 4),
            "flux": round(float(residual[component].sum()), 4),
            "snr": round(peak / sigma, 2),
        })
    if len(detections) > config.max_detections_per_frame:
        detections.sort(key=lambda d: d["snr"], reverse=True)
        detections = detections[:config.max_detections_per_frame]
    return detections, sigma


def load_registered_sequence(registered_dir: Path) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    """Load all successfully registered frames, chronological order."""
    manifest_path = registered_dir / "registration_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} missing — run pipeline.register first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = sorted(name for name, entry in manifest["frames"].items()
                   if entry["status"] in ("ok", "reference"))
    # ponytail: whole sequence in RAM (~600 MB for a full C3 day); switch to
    # a sliding-window loader if multi-day sequences ever appear
    frames, masks = [], []
    for name in names:
        with np.load(registered_dir / f"{name}.npz") as z:
            frames.append(z["data"])
            masks.append(z["valid_mask"])
    return names, frames, masks


def run_motion(names: list[str], frames: list[np.ndarray],
               masks: list[np.ndarray], config: MotionConfig) -> dict:
    """Extract detections for every frame. Returns the detections manifest."""
    per_frame = {}
    for i, name in enumerate(names):
        if config.method == "median":
            if len(frames) < 3:
                raise ValueError("median method needs >= 3 frames")
            background = temporal_background(frames, i, config.window)
        elif config.method == "prev_diff":
            if i == 0:
                per_frame[name] = {"detections": [], "sigma": 0.0,
                                   "note": "no previous frame"}
                continue
            background = frames[i - 1]
        else:
            raise ValueError(f"Unknown method {config.method!r}")
        residual = frames[i] - background
        valid = masks[i] & (masks[i - 1] if config.method == "prev_diff" else masks[i])
        detections, sigma = extract_detections(residual, valid, config)
        per_frame[name] = {"detections": detections, "sigma": round(sigma, 5)}
        logger.info("Frame %s: %d detections (sigma=%.4f)",
                    name, len(detections), sigma)
    return {"config": config.to_dict(), "config_hash": config.config_hash(),
            "frames": per_frame}


def motion_sequence(sequence_id: int, data_root: Path,
                    config: MotionConfig | None = None) -> dict:
    """Run motion extraction over one registered sequence and save results."""
    config = config or MotionConfig()
    registered_dir = data_root / "processed" / "registered" / f"seq_{sequence_id}"
    out_dir = data_root / "processed" / "motion" / f"seq_{sequence_id}"
    names, frames, masks = load_registered_sequence(registered_dir)

    result = run_motion(names, frames, masks, config)
    result["sequence_id"] = sequence_id
    atomic_write_text(out_dir / "detections.json",
                      json.dumps(result, indent=2, sort_keys=True))

    total = sum(len(f["detections"]) for f in result["frames"].values())
    summary = {"sequence_id": sequence_id, "frames": len(names),
               "detections": total,
               "per_frame_mean": round(total / max(1, len(names)), 1)}
    logger.info("Motion summary: %s", summary)
    return summary


def inject_moving_source(frames: list[np.ndarray], start_xy: tuple[float, float],
                         velocity_xy: tuple[float, float], amplitude_sigma: float,
                         sigma_px: float = 1.5) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Add a faint gaussian source moving linearly through copied frames.

    Amplitude is expressed in units of each frame's global robust sigma so
    'faint' means the same thing on C2 and C3. Returns (frames, positions).
    """
    out, positions = [], []
    yy, xx = np.mgrid[0:frames[0].shape[0], 0:frames[0].shape[1]]
    for i, frame in enumerate(frames):
        x = start_xy[0] + velocity_xy[0] * i
        y = start_xy[1] + velocity_xy[1] * i
        amp = amplitude_sigma * robust_sigma(frame[frame != 0])
        blob = amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma_px ** 2))
        out.append((frame + blob).astype(np.float32))
        positions.append((x, y))
    return out, positions


def evaluate_methods(sequence_id: int, data_root: Path) -> None:
    """Synthetic moving-source recovery on real frames, per method.

    A detection within 3 px of the injected position counts as recovered.
    """
    registered_dir = data_root / "processed" / "registered" / f"seq_{sequence_id}"
    names, frames, masks = load_registered_sequence(registered_dir)
    h, w = frames[0].shape
    start, velocity = (w * 0.30, h * 0.70), (4.0, -3.0)

    for amplitude in (10.0, 5.0, 3.0):
        injected, positions = inject_moving_source(frames, start, velocity, amplitude)
        for method in ("median", "prev_diff"):
            config = MotionConfig(method=method)
            result = run_motion(names, injected, masks, config)
            recovered = 0
            for i, name in enumerate(names):
                px, py = positions[i]
                if any(np.hypot(d["x"] - px, d["y"] - py) <= 3.0
                       for d in result["frames"][name]["detections"]):
                    recovered += 1
            total_det = sum(len(f["detections"]) for f in result["frames"].values())
            print(f"amp={amplitude:.0f}sigma method={method:9s} "
                  f"recovered={recovered}/{len(names)} "
                  f"total_detections={total_det}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Temporal motion extraction over a registered sequence")
    parser.add_argument("--sequence-id", type=int, required=True)
    parser.add_argument("--data-root", default=os.environ.get("SOHO_DATA_ROOT", "data"))
    parser.add_argument("--method", choices=("median", "prev_diff"), default="median")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run the synthetic-recovery method comparison")
    args = parser.parse_args(argv)

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.evaluate:
        evaluate_methods(args.sequence_id, Path(args.data_root))
        return 0
    motion_sequence(args.sequence_id, Path(args.data_root),
                    MotionConfig(method=args.method))
    return 0


if __name__ == "__main__":
    sys.exit(main())
