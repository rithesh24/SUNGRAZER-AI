"""Offline unit tests for dataset.events and dataset.labels."""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.events import parse_line, load_events  # noqa: E402
from dataset.labels import (  # noqa: E402
    add_event,
    add_negative_day,
    set_day,
    set_splits,
    validate,
    LABELS_VERSION,
)

SAMPLE_LINES = """\
4968  Jan02,24 18:09:24   H.Tan         C3,C2    Kreutz  Jan02-04,24
4976  Jan29,24 02:18:54   H.Tan           C2     NonGrp  Jan29,24
4978  Feb01,24 12:48:34   HT,JR,MB,EB  C3,C2A,C2 Kreutz  Feb01-03,24
4990* Feb26,24 10:52PM    H.Tan          C3,C2   Kreutz  Feb27,24
5011  Apr30,24 20:53:41   W.Boonplod     C3,C2   Kreutz  Apr30-May01,24
5053  Jun29,24 15:54:12   Z.Xu           C3,C2   Kreutz  Jun29-Jul01,24   cor2
5100  Sep08,24 15:16:44   R.Pickard    C3,C2A,C2 Kreutz  Sep08-12.24
"""


def _fresh_labels() -> dict:
    return {"label_format": LABELS_VERSION, "events": [],
            "negative_days": [], "splits": None}


def test_parse_basic_range():
    e = parse_line(SAMPLE_LINES.splitlines()[0])
    assert e.soho_number == 4968 and not e.flagged
    assert e.cameras == ("C3", "C2") and e.group == "Kreutz"
    assert e.obs_start == date(2024, 1, 2) and e.obs_end == date(2024, 1, 4)
    assert len(e.obs_days) == 3


def test_parse_single_day_and_flag():
    e = parse_line(SAMPLE_LINES.splitlines()[3])
    assert e.flagged and e.obs_start == e.obs_end == date(2024, 2, 27)


def test_parse_cross_month_and_trailing_note():
    e1 = parse_line(SAMPLE_LINES.splitlines()[4])
    assert (e1.obs_start, e1.obs_end) == (date(2024, 4, 30), date(2024, 5, 1))
    e2 = parse_line(SAMPLE_LINES.splitlines()[5])  # trailing "cor2" note
    assert (e2.obs_start, e2.obs_end) == (date(2024, 6, 29), date(2024, 7, 1))


def test_parse_multi_discoverer_c2a():
    e = parse_line(SAMPLE_LINES.splitlines()[2])
    assert e.cameras == ("C3", "C2A", "C2") and e.discoverers == "HT,JR,MB,EB"


def test_malformed_line_returns_none_and_is_reported():
    assert parse_line(SAMPLE_LINES.splitlines()[6]) is None  # "Sep08-12.24" typo
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "conf.txt"
        path.write_text(SAMPLE_LINES, encoding="utf-8")
        events, failures = load_events(path)
        assert len(events) == 6 and len(failures) == 1


def test_labels_event_lifecycle():
    with tempfile.TemporaryDirectory() as tmp:
        conf = Path(tmp) / "conf_2024.txt"
        conf.write_text(SAMPLE_LINES, encoding="utf-8")
        labels = _fresh_labels()
        record = add_event(labels, 4968, conf)
        assert record["status"] == "selected"
        assert sorted(record["days"]) == ["2024-01-02", "2024-01-03", "2024-01-04"]
        set_day(labels, 4968, "2024-01-03", sequence_id=3, track_ids=[])
        assert labels["events"][0]["status"] == "ingested"
        set_day(labels, 4968, "2024-01-03", sequence_id=None,
                track_ids=["seq3_abc_00001"])
        assert labels["events"][0]["status"] == "labeled"
        assert validate(labels) == []  # no data_root: file checks skipped


def test_negative_day_overlap_detected():
    with tempfile.TemporaryDirectory() as tmp:
        conf = Path(tmp) / "conf.txt"
        conf.write_text(SAMPLE_LINES, encoding="utf-8")
        labels = _fresh_labels()
        add_event(labels, 4968, conf)
        add_negative_day(labels, "2024-01-01", "C3", 1, ["conf_2024.txt"])
        assert validate(labels) == []
        add_negative_day(labels, "2024-01-03", "C3", 9, ["conf_2024.txt"])
        assert any("overlaps" in p for p in validate(labels))


def test_adjacent_days_in_different_splits_flagged():
    labels = _fresh_labels()
    for day in ("2024-01-01", "2024-01-02", "2024-06-09"):
        add_negative_day(labels, day, "C3", 1, ["conf.txt"])
    labels["splits"] = {"train": ["2024-01-01"],
                        "test": ["2024-01-02", "2024-06-09"]}
    assert any("leakage" in p for p in validate(labels))
    labels["splits"] = {"train": ["2024-01-01", "2024-01-02"],
                        "test": ["2024-06-09"]}
    assert validate(labels) == []


def test_split_day_without_data_flagged():
    labels = _fresh_labels()
    add_negative_day(labels, "2024-01-01", "C3", 1, ["conf.txt"])
    set_splits(labels, {"train": ["2024-01-01"], "test": ["2024-03-15"]})
    assert any("no dataset data" in p for p in validate(labels))


def test_data_day_missing_from_splits_flagged():
    labels = _fresh_labels()
    add_negative_day(labels, "2024-01-01", "C3", 1, ["conf.txt"])
    add_negative_day(labels, "2024-01-10", "C3", 2, ["conf.txt"])
    set_splits(labels, {"train": ["2024-01-01"]})
    assert any("no split assignment" in p for p in validate(labels))


def test_day_in_two_splits_flagged():
    labels = _fresh_labels()
    add_negative_day(labels, "2024-01-01", "C3", 1, ["conf.txt"])
    set_splits(labels, {"train": ["2024-01-01"], "test": ["2024-01-01"]})
    assert any("in both splits" in p for p in validate(labels))


def test_labeled_but_not_ingested_is_invalid():
    labels = _fresh_labels()
    labels["events"].append({
        "soho_number": 9999, "group": "Kreutz", "cameras": ["C3"],
        "obs_start": "2024-03-01", "obs_end": "2024-03-01",
        "source_file": "x", "source_line": "x", "status": "labeled",
        "days": {"2024-03-01": {"sequence_id": None,
                                "positive_track_ids": ["seq9_x_00000"]}},
        "notes": ""})
    assert any("labeled but not ingested" in p for p in validate(labels))


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
