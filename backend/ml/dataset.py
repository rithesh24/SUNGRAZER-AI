"""PyTorch dataset over labeled track crop stacks (tracker section 14).

The classified unit is a track (dataset.labels docstring): item = one
track's temporal crop stack + binary label (1 = confirmed comet track).

Split resolution (labels.json, day-level splits):

- A split's sequences come from its days: an event day contributes its
  ``sequence_id`` plus the sequence of every positive track's ``seq<id>_``
  prefix (a day may span several sequences); a negative day contributes
  its ``sequence_id``.
- Positives: the ``positive_track_ids`` of the split's event days.
- Negatives: every other extracted track of those sequences — EXCEPT all
  tracks of an event day with no positive tracks (e.g. 2024-01-02: a
  confirmed comet is present below the detection floor, so none of that
  day's tracks may be trusted as a negative).

Negatives from positive days are *unverified* (only negative days are
list-checked); contamination risk is documented in progress.md session 13.

Tensorization (``TrackCropDataset.__getitem__``):

- crops ``[T, K, K]`` -> normalized ``float32 [max_frames, 1, K, K]``:
  ``asinh(x / sigma)`` with the stack's robust MAD sigma — zero-centered
  residual noise maps to ~unit scale, faint sources stay linear, saturated
  planets/bleed are compressed instead of dominating the loss.
- Tracks longer than ``max_frames`` are subsampled with a deterministic
  uniform stride (reproducible evaluation); shorter tracks are zero-padded
  and the true length is returned so the temporal model can mask padding.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline.motion import robust_sigma

logger = logging.getLogger(__name__)

_SEQ_PREFIX = re.compile(r"seq(\d+)_")


@dataclass(frozen=True)
class SplitTracks:
    """Resolved track membership for one split."""

    positives: list[str]
    negatives: list[str]

    @property
    def all_tracks(self) -> list[str]:
        return self.positives + self.negatives


def _sequence_of(track_id: str) -> int:
    match = _SEQ_PREFIX.match(track_id)
    if not match:
        raise ValueError(f"track id {track_id!r} lacks a seq<id>_ prefix")
    return int(match.group(1))


def resolve_split(labels: dict, split: str, data_root: Path) -> SplitTracks:
    """Resolve a split name to positive/negative track-ID lists."""
    if not labels.get("splits") or split not in labels["splits"]:
        raise ValueError(f"split {split!r} not defined in labels")
    split_days = set(labels["splits"][split])

    positives: list[str] = []
    sequences: set[int] = set()
    excluded_sequences: set[int] = set()
    for event in labels["events"]:
        for day, rec in event["days"].items():
            if day not in split_days or rec["sequence_id"] is None:
                continue
            day_sequences = {rec["sequence_id"]}
            day_sequences |= {_sequence_of(t) for t in rec["positive_track_ids"]}
            sequences |= day_sequences
            if rec["positive_track_ids"]:
                positives.extend(rec["positive_track_ids"])
            else:
                # Comet present (per the confirmation list) but below the
                # detection floor: no track of this day is a safe negative.
                excluded_sequences |= day_sequences
                logger.info("split %s: excluding seq %s from the negative "
                            "pool (unlabeled comet day %s)",
                            split, sorted(day_sequences), day)
    for neg in labels["negative_days"]:
        if neg["day"] in split_days:
            sequences.add(neg["sequence_id"])

    positive_set = set(positives)
    negatives: list[str] = []
    for seq_id in sorted(sequences - excluded_sequences):
        manifest_path = (data_root / "processed" / "candidate_crops"
                         / f"seq_{seq_id}" / "crops_manifest.json")
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} missing — run `python -m ml.crops "
                f"--sequence-id {seq_id}` first")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name in manifest["track_files"]:
            track_id = name.removesuffix(".npz")
            if track_id not in positive_set:
                negatives.append(track_id)
    missing = [t for t in positives
               if not (data_root / "processed" / "candidate_crops"
                       / f"seq_{_sequence_of(t)}" / f"{t}.npz").exists()]
    if missing:
        raise FileNotFoundError(f"positive crop stacks missing: {missing}")
    return SplitTracks(positives=positives, negatives=negatives)


class TrackCropDataset(Dataset):
    """Binary track-classification dataset for one split."""

    def __init__(self, data_root: Path, split: str,
                 max_frames: int = 16, sigma_floor: float = 1e-6):
        self.data_root = Path(data_root)
        self.split = split
        self.max_frames = max_frames
        self.sigma_floor = sigma_floor
        labels_file = self.data_root / "dataset" / "v1" / "labels.json"
        labels = json.loads(labels_file.read_text(encoding="utf-8"))
        tracks = resolve_split(labels, split, self.data_root)
        self.track_ids = tracks.all_tracks
        self.labels = torch.cat([torch.ones(len(tracks.positives)),
                                 torch.zeros(len(tracks.negatives))])
        logger.info("split %s: %d positives, %d negatives",
                    split, len(tracks.positives), len(tracks.negatives))

    def __len__(self) -> int:
        return len(self.track_ids)

    def _load_stack(self, track_id: str) -> np.ndarray:
        path = (self.data_root / "processed" / "candidate_crops"
                / f"seq_{_sequence_of(track_id)}" / f"{track_id}.npz")
        with np.load(path) as z:
            return z["crops"]

    def __getitem__(self, index: int) -> dict:
        track_id = self.track_ids[index]
        stack = self._load_stack(track_id)
        sigma = max(robust_sigma(stack.ravel()), self.sigma_floor)
        stack = np.arcsinh(stack / sigma)

        n = stack.shape[0]
        if n > self.max_frames:  # deterministic uniform subsample
            indices = np.linspace(0, n - 1, self.max_frames).round().astype(int)
            stack = stack[indices]
            n = self.max_frames
        padded = np.zeros((self.max_frames, *stack.shape[1:]), dtype=np.float32)
        padded[:n] = stack
        return {
            "crops": torch.from_numpy(padded).unsqueeze(1),  # [T, 1, K, K]
            "length": torch.tensor(n, dtype=torch.long),
            "label": self.labels[index],
            "track_id": track_id,
        }
