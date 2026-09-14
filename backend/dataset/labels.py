"""Versioned label store for the candidate dataset (tracker section 12).

Usage:
    python -m dataset.labels init
    python -m dataset.labels add-event --soho 4968 --conf-file PATH
    python -m dataset.labels add-negative-day --day 2024-01-01 --camera C3 \
        --sequence-id 1 --verified-against conf_2023.txt,conf_2024.txt
    python -m dataset.labels set-day --soho 4968 --day 2024-01-03 \
        --sequence-id 3 [--tracks id1,id2]
    python -m dataset.labels validate

Label file: ``data/dataset/v1/labels.json`` (``labels_v1``):

- ``events``       — selected known-comet events (from Sungrazer
  confirmation lists; see dataset.events). Per event: identity
  (soho_number, group, cameras, obs dates, source file), ``status``
  (``selected`` -> ``ingested`` -> ``labeled``) and per-day records:
  ``sequence_id`` (once ingested) and ``positive_track_ids`` — track IDs
  from ``processed/tracks/seq_<id>/tracks.json`` confirmed to be the
  comet. A track ID is the positive label unit: the ranking model
  classifies tracks.
- ``negative_days`` — full days verified comet-free against the
  confirmation lists; every kept track of those sequences is a negative.
  The lists are not exhaustive (NRL's own caveat), so a negative day
  records exactly which lists it was checked against.

Leakage rule (claude.md section 18, enforced at split time): all
sequences from the same or adjacent calendar days must land in the same
split — star fields barely move day-to-day. Splits are stored here once
more than one event exists.

``validate`` re-checks the whole file: date formats, status values,
day-within-observation-window, no day that is both negative and part of
a positive event for the same camera, and — when the processed data is
on disk — that every referenced sequence directory and positive track ID
actually exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

from dataset.events import load_events
from pipeline.fileio import atomic_write_text

logger = logging.getLogger(__name__)

LABELS_VERSION = "labels_v1"
DATASET_DIR = "dataset/v1"
_STATUSES = ("selected", "ingested", "labeled")


def labels_path(data_root: Path) -> Path:
    return data_root / DATASET_DIR / "labels.json"


def load_labels(data_root: Path) -> dict:
    path = labels_path(data_root)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python -m dataset.labels init`")
    return json.loads(path.read_text(encoding="utf-8"))


def save_labels(data_root: Path, labels: dict) -> None:
    atomic_write_text(labels_path(data_root),
                      json.dumps(labels, indent=2, sort_keys=True))


def init_labels(data_root: Path) -> dict:
    path = labels_path(data_root)
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    labels = {"label_format": LABELS_VERSION, "events": [],
              "negative_days": [], "splits": None}
    save_labels(data_root, labels)
    return labels


def add_event(labels: dict, soho_number: int, conf_file: Path) -> dict:
    """Add one confirmed comet event from a confirmation-list file."""
    if any(e["soho_number"] == soho_number for e in labels["events"]):
        raise ValueError(f"SOHO-{soho_number} already in labels")
    events, _ = load_events(conf_file)
    matches = [e for e in events if e.soho_number == soho_number]
    if len(matches) != 1:
        raise ValueError(f"SOHO-{soho_number}: {len(matches)} matches in {conf_file}")
    event = matches[0]
    record = {
        "soho_number": event.soho_number,
        "group": event.group,
        "cameras": list(event.cameras),
        "obs_start": event.obs_start.isoformat(),
        "obs_end": event.obs_end.isoformat(),
        "source_file": conf_file.name,
        "source_line": event.source_line,
        "status": "selected",
        "days": {d.isoformat(): {"sequence_id": None, "positive_track_ids": []}
                 for d in event.obs_days},
        "notes": "",
    }
    labels["events"].append(record)
    return record


def add_negative_day(labels: dict, day: str, camera: str, sequence_id: int,
                     verified_against: list[str]) -> dict:
    date.fromisoformat(day)  # validates format
    if any(n["day"] == day and n["camera"] == camera
           for n in labels["negative_days"]):
        raise ValueError(f"{day}/{camera} already a negative day")
    record = {"day": day, "camera": camera.upper(), "sequence_id": sequence_id,
              "verified_against": verified_against}
    labels["negative_days"].append(record)
    return record


def set_day(labels: dict, soho_number: int, day: str,
            sequence_id: int | None, track_ids: list[str]) -> None:
    event = next((e for e in labels["events"]
                  if e["soho_number"] == soho_number), None)
    if event is None:
        raise ValueError(f"SOHO-{soho_number} not in labels")
    if day not in event["days"]:
        raise ValueError(f"{day} outside observation window of SOHO-{soho_number}")
    if sequence_id is not None:
        event["days"][day]["sequence_id"] = sequence_id
    if track_ids:
        event["days"][day]["positive_track_ids"] = track_ids
    ingested = [d for d in event["days"].values() if d["sequence_id"] is not None]
    labeled = [d for d in event["days"].values() if d["positive_track_ids"]]
    event["status"] = ("labeled" if labeled else
                       "ingested" if ingested else "selected")


def validate(labels: dict, data_root: Path | None = None) -> list[str]:
    """Full consistency check; returns problems (empty = valid)."""
    problems = []
    if labels.get("label_format") != LABELS_VERSION:
        problems.append(f"unknown label_format {labels.get('label_format')!r}")

    positive_days: set[tuple[str, str]] = set()
    for event in labels.get("events", []):
        ident = f"SOHO-{event.get('soho_number')}"
        try:
            start = date.fromisoformat(event["obs_start"])
            end = date.fromisoformat(event["obs_end"])
        except (KeyError, ValueError) as exc:
            problems.append(f"{ident}: bad obs dates ({exc})")
            continue
        if event.get("status") not in _STATUSES:
            problems.append(f"{ident}: bad status {event.get('status')!r}")
        for day_str, day_rec in event.get("days", {}).items():
            day = date.fromisoformat(day_str)
            if not start <= day <= end:
                problems.append(f"{ident}: day {day_str} outside window")
            for camera in event.get("cameras", []):
                positive_days.add((day_str, camera.upper().rstrip("A")))
            if day_rec["positive_track_ids"] and day_rec["sequence_id"] is None:
                problems.append(f"{ident}: {day_str} labeled but not ingested")
            if data_root and day_rec["sequence_id"] is not None:
                tracks_file = (data_root / "processed" / "tracks"
                               / f"seq_{day_rec['sequence_id']}" / "tracks.json")
                if not tracks_file.exists():
                    problems.append(f"{ident}: {tracks_file} missing")
                # A day can span several sequences (mid-day observation gap
                # splits it); each track ID names its own sequence
                # ("seq<id>_..."), so resolve per track, not per day.
                for tid in day_rec["positive_track_ids"]:
                    match = re.match(r"seq(\d+)_", tid)
                    if not match:
                        problems.append(f"{ident}: track id {tid} lacks a "
                                        f"seq<id>_ prefix")
                        continue
                    tid_file = (data_root / "processed" / "tracks"
                                / f"seq_{match.group(1)}" / "tracks.json")
                    if not tid_file.exists():
                        problems.append(f"{ident}: {tid_file} missing")
                        continue
                    known = {t["track_id"] for t in json.loads(
                        tid_file.read_text(encoding="utf-8"))["tracks"]}
                    if tid not in known:
                        problems.append(f"{ident}: track {tid} not in {tid_file}")

    for neg in labels.get("negative_days", []):
        key = (neg["day"], neg["camera"].upper())
        if key in positive_days:
            problems.append(f"negative day {key} overlaps a positive event")
        if not neg.get("verified_against"):
            problems.append(f"negative day {neg['day']}: no verification sources")

    # Adjacent positive/negative days are allowed but must share a split.
    if labels.get("splits"):
        day_split: dict[str, str] = {}
        for split, days in labels["splits"].items():
            for d in days:
                day_split[d] = split
        for d, split in day_split.items():
            neighbor = (date.fromisoformat(d) + timedelta(days=1)).isoformat()
            if neighbor in day_split and day_split[neighbor] != split:
                problems.append(
                    f"adjacent days {d}/{neighbor} in different splits "
                    f"({split}/{day_split[neighbor]}) — leakage")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dataset label store")
    parser.add_argument("--data-root", default=os.environ.get("SOHO_DATA_ROOT", "data"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    p_add = sub.add_parser("add-event")
    p_add.add_argument("--soho", type=int, required=True)
    p_add.add_argument("--conf-file", type=Path, required=True)
    p_neg = sub.add_parser("add-negative-day")
    p_neg.add_argument("--day", required=True)
    p_neg.add_argument("--camera", required=True)
    p_neg.add_argument("--sequence-id", type=int, required=True)
    p_neg.add_argument("--verified-against", required=True,
                       help="comma-separated confirmation-list files checked")
    p_day = sub.add_parser("set-day")
    p_day.add_argument("--soho", type=int, required=True)
    p_day.add_argument("--day", required=True)
    p_day.add_argument("--sequence-id", type=int)
    p_day.add_argument("--tracks", default="",
                       help="comma-separated positive track ids")
    sub.add_parser("validate")
    args = parser.parse_args(argv)

    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
    data_root = Path(args.data_root)

    if args.command == "init":
        init_labels(data_root)
        print(f"created {labels_path(data_root)}")
        return 0
    labels = load_labels(data_root)
    if args.command == "add-event":
        record = add_event(labels, args.soho, args.conf_file)
        print(f"added SOHO-{record['soho_number']} "
              f"({record['obs_start']}..{record['obs_end']})")
    elif args.command == "add-negative-day":
        add_negative_day(labels, args.day, args.camera, args.sequence_id,
                         args.verified_against.split(","))
        print(f"added negative day {args.day}/{args.camera}")
    elif args.command == "set-day":
        tracks = [t for t in args.tracks.split(",") if t]
        set_day(labels, args.soho, args.day, args.sequence_id, tracks)
        print(f"updated SOHO-{args.soho} {args.day}")
    elif args.command == "validate":
        problems = validate(labels, data_root)
        for p in problems:
            logger.error(p)
        print("VALID" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    save_labels(data_root, labels)
    problems = validate(labels, data_root)
    if problems:
        for p in problems:
            logger.error(p)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
