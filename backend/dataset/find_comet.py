"""Shortlist comet-candidate tracks of a sequence for positive labeling.

Usage:
    python -m dataset.find_comet --sequence-id N [--top 10]

Ranks a sequence's tracks (from Motion DNA features) by how strongly they
match the sungrazer signature and prints a shortlist for human
verification. This tool NEVER labels anything automatically — a track
becomes a positive only after visual confirmation against the archive
imagery (``dataset.labels set-day --tracks ...``).

Hard requirements (anything failing these is not shown):
- sunward motion: ``radial_speed_px_s`` < 0 — the defining Kreutz property;
- persistence: ``n_frames`` >= ``--min-frames`` (default 8).

Ranking score (higher = more comet-like), a transparent sum of:
- direction consistency (0..1);
- smoothness: 1 / (1 + linear_rms_px);
- speed above star drift: min(speed / (3 * star drift), 1) — LASCO stars
  drift ~0.0009 px/s (measured, progress.md session 7); a sungrazer in C3
  moves several times that;
- sunward fraction: |radial_speed| / speed (1 = motion purely sunward).

The score is a triage heuristic, not a detector — the ranking model
(techspec section 16) replaces it once trained.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

STAR_DRIFT_PX_S = 0.0009  # measured C3 star drift (progress.md 2026-09-13 (7))


def comet_score(f: dict) -> float | None:
    """Signature score for one Motion DNA feature record; None = excluded."""
    radial = f.get("radial_speed_px_s")
    speed = f.get("speed_mean_px_s") or 0.0
    if radial is None or radial >= 0 or speed <= 0:
        return None
    consistency = f.get("direction_consistency") or 0.0
    rms = f.get("linear_rms_px")
    smoothness = 1.0 / (1.0 + rms) if rms is not None else 0.0
    speed_factor = min(speed / (3 * STAR_DRIFT_PX_S), 1.0)
    sunward_fraction = min(abs(radial) / speed, 1.0)
    return consistency + smoothness + speed_factor + sunward_fraction


def shortlist(sequence_id: int, data_root: Path, min_frames: int,
              top: int) -> list[tuple[float, str, dict]]:
    """(score, track_id, features) for the top comet-like tracks."""
    dna_path = (data_root / "processed" / "motion_dna"
                / f"seq_{sequence_id}" / "motion_dna.json")
    if not dna_path.exists():
        raise FileNotFoundError(f"{dna_path} missing — run pipeline.motion_dna first")
    dna = json.loads(dna_path.read_text(encoding="utf-8"))
    scored = []
    for track_id, f in dna["features"].items():
        if f["n_frames"] < min_frames:
            continue
        score = comet_score(f)
        if score is not None:
            scored.append((score, track_id, f))
    scored.sort(reverse=True)
    return scored[:top]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank a sequence's tracks by sungrazer signature")
    parser.add_argument("--sequence-id", type=int, required=True)
    parser.add_argument("--data-root", default=os.environ.get("SOHO_DATA_ROOT", "data"))
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)

    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
    results = shortlist(args.sequence_id, Path(args.data_root),
                        args.min_frames, args.top)
    if not results:
        print("No sunward tracks matched — check min-frames or the sequence.")
        return 1
    for score, track_id, f in results:
        print(f"score={score:.3f}  {track_id}  n={f['n_frames']:3d} "
              f"speed={f['speed_mean_px_s']:.5f}px/s "
              f"radial={f['radial_speed_px_s']:.5f} "
              f"consist={f['direction_consistency']} rms={f['linear_rms_px']} "
              f"r={f['r_min_px']}..{f['r_max_px']}px flux~{f['flux_mean']:.0f}")
    print("\nVerify visually before labeling: "
          "python -m dataset.labels set-day --soho N --day D --tracks <id>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
