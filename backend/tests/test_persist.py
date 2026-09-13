"""Offline unit tests for manifest -> PostgreSQL persistence (no DB required).

Run:  python tests/test_persist.py   (or pytest tests/)

The full write path was validated against live PostgreSQL 17 (see
docs/progress.md session 2026-09-13); these tests pin the pure mapping and
parsing logic.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.persist import image_row_values, parse_date_obs


def test_parse_date_obs_with_fraction():
    parsed = parse_date_obs("2024/01/01T00:06:06.030")
    assert parsed == datetime(2024, 1, 1, 0, 6, 6, 30000, tzinfo=timezone.utc), parsed


def test_parse_date_obs_without_fraction():
    parsed = parse_date_obs("2024/01/01T12:30:00")
    assert parsed == datetime(2024, 1, 1, 12, 30, tzinfo=timezone.utc), parsed


def test_parse_date_obs_date_only():
    parsed = parse_date_obs("2024/01/01")
    assert parsed == datetime(2024, 1, 1, tzinfo=timezone.utc), parsed


def test_parse_date_obs_bad_input():
    assert parse_date_obs(None) is None
    assert parse_date_obs("") is None
    assert parse_date_obs("not-a-date") is None


def test_image_row_values_complete_entry(tmp_path=None):
    entry = {
        "status": "complete",
        "source_url": "https://example.org/lz/level_05/240101/c3/32765039.fts",
        "sha256": "ab" * 32,
        "size_bytes": 2108160,
        "date_obs": "2024/01/01T00:06:06.030",
    }
    # Nonexistent raw dir: dimensions must degrade to None, not raise.
    values = image_row_values(
        "240101/c3/32765039.fts", "240101", "c3", "32765039.fts",
        entry, Path("nonexistent-raw-dir"),
    )
    assert values["source_identifier"] == "240101/c3/32765039.fts"
    assert values["local_path"] == "raw/soho/lasco/240101/c3/32765039.fts"
    assert values["instrument"] == "LASCO/C3"
    assert values["checksum"] == "ab" * 32
    assert values["size_bytes"] == 2108160
    assert values["ingestion_status"] == "complete"
    assert values["error_message"] is None
    assert values["width"] is None and values["height"] is None
    assert values["observation_time"].year == 2024


def test_image_row_values_failed_entry():
    entry = {
        "status": "failed",
        "source_url": "https://example.org/x.fts",
        "error": "HTTP 500",
    }
    values = image_row_values("240101/c2/x.fts", "240101", "c2", "x.fts",
                              entry, Path("nonexistent-raw-dir"))
    assert values["ingestion_status"] == "failed"
    assert values["error_message"] == "HTTP 500"
    assert values["instrument"] == "LASCO/C2"
    assert values["checksum"] is None
    assert values["observation_time"] is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("All persist tests passed.")
