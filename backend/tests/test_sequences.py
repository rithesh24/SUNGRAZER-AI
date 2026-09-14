"""Offline unit tests for sequence grouping (no DB required).

Run:  python tests/test_sequences.py   (or pytest tests/)

The DB-backed attach path was validated against live PostgreSQL 17 (see
docs/progress.md session 2026-09-13): 3 C3 + 3 C2 frames produced two
per-instrument sequences with correct windows, idempotent on re-run, and
check_sequences reported no problems.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.sequences import group_by_gap

T0 = datetime(2024, 1, 1, 0, 0, 0)
GAP = timedelta(minutes=60)


def minutes(*offsets):
    return [T0 + timedelta(minutes=m) for m in offsets]


def test_empty():
    assert group_by_gap([], GAP) == []


def test_single_frame():
    assert group_by_gap(minutes(0), GAP) == [[0]]


def test_regular_cadence_one_group():
    # ~12-min C3 cadence: everything within one sequence
    assert group_by_gap(minutes(0, 12, 24, 36), GAP) == [[0, 1, 2, 3]]


def test_split_on_large_gap():
    # 4-hour outage between frame 2 and 3 splits the day into two sequences
    assert group_by_gap(minutes(0, 12, 24, 264, 276), GAP) == [[0, 1, 2], [3, 4]]


def test_gap_boundary_inclusive():
    # exactly the threshold stays in one group; one second over splits
    assert group_by_gap(minutes(0, 60), GAP) == [[0, 1]]
    over = [T0, T0 + GAP + timedelta(seconds=1)]
    assert group_by_gap(over, GAP) == [[0], [1]]


def test_multiple_splits():
    times = minutes(0, 12, 120, 132, 300)
    assert group_by_gap(times, GAP) == [[0, 1], [2, 3], [4]]


def test_split_on_utc_day_boundary():
    # 23:54 -> next-day 00:06 is only 12 min apart but must split:
    # sequences never cross a UTC day (dataset labels are per day)
    times = [datetime(2024, 1, 1, 23, 42), datetime(2024, 1, 1, 23, 54),
             datetime(2024, 1, 2, 0, 6), datetime(2024, 1, 2, 0, 18)]
    assert group_by_gap(times, GAP) == [[0, 1], [2, 3]]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("All sequence tests passed.")
