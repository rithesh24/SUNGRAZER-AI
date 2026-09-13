"""Image registration: align preprocessed frames to a sequence reference.

Usage:
    python -m pipeline.register --sequence-id N [--data-root PATH]
    python -m pipeline.register --sequence-id N --evaluate

Registers every preprocessed frame of a sequence to the sequence's reference
frame (the first QA-usable one) by translation, estimated with windowed
phase correlation. Direct-to-reference avoids the drift accumulation of
chained frame-to-frame registration; SOHO pointing keeps residual shifts
small, so a common reference stays valid across a sequence.

Quality and failure handling (techspec section 10): every frame records its
estimated (dx, dy) and the phase-correlation response (peak sharpness,
0..1). Estimates with response below ``min_response`` or shift above
``max_shift_px`` are recorded as FAILED and the frame is written
UNSHIFTED — a poor registration is never silently applied.

``--evaluate`` runs a synthetic-shift recovery experiment on real frames of
the sequence, comparing phase correlation against ECC translation, and
prints per-method errors. Results that guided the baseline choice are
recorded in docs/progress.md (session 2026-09-13).

Known failure modes:
- The static field mask (occulter/border, zeroed pixels) is a strong
  zero-shift feature; a real shift larger than a few pixels is biased low.
  Acceptable for LASCO's sub-pixel jitter; revisit if shifts ever grow.
- Frames dominated by noise (weak structure) yield low response and are
  correctly flagged FAILED rather than mis-registered.
- cv2.phaseCorrelate sub-pixel estimates are biased toward integer shifts:
  measured worst case ~0.3 px error at exactly half-pixel true shifts,
  ~0.01 px at integer/quarter-pixel shifts (see tests). Upgrade path if
  measured LASCO jitter lands in that range: upsampled phase correlation
  (scikit-image phase_cross_correlation).
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

from pipeline.fileio import atomic_write_bytes_via, atomic_write_text

logger = logging.getLogger(__name__)

REGISTER_VERSION = "reg_v1"

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_REFERENCE = "reference"


@dataclass(frozen=True)
class RegisterConfig:
    """Deterministic configuration for one registration run."""

    version: str = REGISTER_VERSION
    method: str = "phase_correlation"
    min_response: float = 0.05      # below: estimate untrustworthy
    max_shift_px: float = 10.0      # LASCO jitter is sub-pixel; more = bogus

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def estimate_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float, float]:
    """Estimate (dx, dy) moving the frame onto the reference, with response.

    Windowed phase correlation on float32 data. Returns the shift to APPLY
    to ``moving`` so it aligns with ``reference`` (cv2.phaseCorrelate
    reports the displacement of ``moving`` relative to ``reference``, so we
    negate it), plus the correlation response in [0, 1].
    """
    window = cv2.createHanningWindow(reference.shape[::-1], cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(
        reference.astype(np.float32), moving.astype(np.float32), window
    )
    return -dx, -dy, float(response)


def apply_shift(data: np.ndarray, valid_mask: np.ndarray,
                dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Translate data (linear interp) and mask (nearest); borders invalid."""
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    size = (data.shape[1], data.shape[0])
    shifted = cv2.warpAffine(data, matrix, size, flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    mask = cv2.warpAffine(valid_mask.astype(np.uint8), matrix, size,
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return shifted.astype(np.float32), mask.astype(bool)


def register_frame(reference: np.ndarray, data: np.ndarray,
                   valid_mask: np.ndarray, config: RegisterConfig) -> dict:
    """Register one frame; returns {data, valid_mask, dx, dy, response, status}."""
    dx, dy, response = estimate_shift(reference, data)
    shift_mag = float(np.hypot(dx, dy))
    if response < config.min_response or shift_mag > config.max_shift_px:
        logger.warning("Registration FAILED (response=%.4f shift=%.2fpx) — "
                       "frame kept unshifted", response, shift_mag)
        return {"data": data, "valid_mask": valid_mask, "dx": 0.0, "dy": 0.0,
                "measured_dx": dx, "measured_dy": dy,
                "response": response, "status": STATUS_FAILED}
    shifted, mask = apply_shift(data, valid_mask, dx, dy)
    return {"data": shifted, "valid_mask": mask, "dx": dx, "dy": dy,
            "response": response, "status": STATUS_OK}


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        return z["data"], z["valid_mask"]


def _save_npz(path: Path, data: np.ndarray, valid_mask: np.ndarray) -> None:
    atomic_write_bytes_via(
        path, lambda fh: np.savez_compressed(fh, data=data, valid_mask=valid_mask))


def usable_frames(normalized_dir: Path) -> list[str]:
    """Chronological stems of QA-usable frames per the preprocess manifest."""
    manifest_path = normalized_dir / "preprocess_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} missing — run pipeline.preprocess first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return sorted(name for name, entry in manifest["frames"].items()
                  if entry.get("ok") and entry.get("qa", {}).get("usable"))


def register_sequence(sequence_id: int, data_root: Path,
                      config: RegisterConfig | None = None) -> dict:
    """Register all usable preprocessed frames of one sequence to disk.

    Writes ``processed/registered/seq_<id>/<stem>.npz`` plus
    ``registration_manifest.json`` (config + per-frame shift/response/
    status). Idempotent per config hash.
    """
    config = config or RegisterConfig()
    normalized_dir = data_root / "processed" / "normalized" / f"seq_{sequence_id}"
    out_dir = data_root / "processed" / "registered" / f"seq_{sequence_id}"
    names = usable_frames(normalized_dir)
    if len(names) < 2:
        raise ValueError(f"sequence {sequence_id}: need >= 2 usable frames, "
                         f"have {len(names)}")

    manifest_path = out_dir / "registration_manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {"frames": {}})
    if manifest.get("config_hash") != config.config_hash():
        manifest = {"frames": {}}
    manifest.update(config=config.to_dict(), config_hash=config.config_hash(),
                    sequence_id=sequence_id, reference=names[0])

    reference_data, reference_mask = _load_npz(normalized_dir / f"{names[0]}.npz")
    if manifest["frames"].get(names[0], {}).get("status") != STATUS_REFERENCE:
        _save_npz(out_dir / f"{names[0]}.npz", reference_data, reference_mask)
        manifest["frames"][names[0]] = {"dx": 0.0, "dy": 0.0, "response": 1.0,
                                        "status": STATUS_REFERENCE}

    registered = skipped = failed = 0
    for name in names[1:]:
        if name in manifest["frames"]:
            skipped += 1
            continue
        data, mask = _load_npz(normalized_dir / f"{name}.npz")
        if data.shape != reference_data.shape:
            # LASCO days mix in occasional 2x2-binned (512x512) frames whose
            # plate scale differs; they are excluded, not resized — cross-
            # binning registration is a documented later extension.
            logger.warning("Frame %s shape %s != reference %s — excluded",
                           name, data.shape, reference_data.shape)
            manifest["frames"][name] = {
                "dx": 0.0, "dy": 0.0, "response": 0.0,
                "status": STATUS_FAILED,
                "reason": f"shape {list(data.shape)} != reference "
                          f"{list(reference_data.shape)}",
            }
            failed += 1
            atomic_write_text(manifest_path,
                              json.dumps(manifest, indent=2, sort_keys=True))
            continue
        result = register_frame(reference_data, data, mask, config)
        _save_npz(out_dir / f"{name}.npz", result["data"], result["valid_mask"])
        manifest["frames"][name] = {k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in result.items()
                                    if k not in ("data", "valid_mask")}
        if result["status"] == STATUS_OK:
            registered += 1
        else:
            failed += 1
        atomic_write_text(manifest_path,
                          json.dumps(manifest, indent=2, sort_keys=True))

    summary = {"sequence_id": sequence_id, "frames": len(names),
               "registered": registered, "skipped": skipped, "failed": failed}
    logger.info("Registration summary: %s", summary)
    return summary


def evaluate_methods(sequence_id: int, data_root: Path) -> list[dict]:
    """Synthetic-shift recovery experiment on a real frame of the sequence.

    Applies known sub-pixel and multi-pixel shifts to a real preprocessed
    frame and measures how accurately phase correlation and ECC translation
    recover them. Prints a table and returns the rows.
    """
    normalized_dir = data_root / "processed" / "normalized" / f"seq_{sequence_id}"
    names = usable_frames(normalized_dir)
    data, mask = _load_npz(normalized_dir / f"{names[0]}.npz")

    def ecc_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float]:
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
        try:
            cv2.findTransformECC(reference, moving, warp,
                                 cv2.MOTION_TRANSLATION, criteria)
        except cv2.error:
            return float("nan"), float("nan")
        return float(warp[0, 2]), float(warp[1, 2])

    rows = []
    for true_dx, true_dy in [(0.0, 0.0), (0.5, -0.3), (2.3, 1.7), (5.1, -4.2)]:
        shifted, _ = apply_shift(data, mask, true_dx, true_dy)
        pc_dx, pc_dy, response = estimate_shift(data, shifted)
        ecc_dx, ecc_dy = ecc_shift(data, shifted)
        rows.append({
            "true": (true_dx, true_dy),
            "phase_corr_err": round(float(np.hypot(pc_dx + true_dx, pc_dy + true_dy)), 4),
            "phase_corr_response": round(response, 4),
            "ecc_err": round(float(np.hypot(ecc_dx + true_dx, ecc_dy + true_dy)), 4),
        })
        print(f"true=({true_dx:+.1f},{true_dy:+.1f})  "
              f"phase_corr_err={rows[-1]['phase_corr_err']:.4f}px "
              f"(response={response:.3f})  ecc_err={rows[-1]['ecc_err']:.4f}px")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Register preprocessed frames of one sequence")
    parser.add_argument("--sequence-id", type=int, required=True)
    parser.add_argument("--data-root", default=os.environ.get("SOHO_DATA_ROOT", "data"))
    parser.add_argument("--evaluate", action="store_true",
                        help="Run the method-comparison experiment instead")
    args = parser.parse_args(argv)

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.evaluate:
        evaluate_methods(args.sequence_id, Path(args.data_root))
        return 0
    summary = register_sequence(args.sequence_id, Path(args.data_root))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
