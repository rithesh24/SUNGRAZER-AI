"""Offline tests for the candidate loader label logic and evidence attribution."""

from dataset.candidates import candidate_rows, label_maps  # noqa: F401  (import check)
from dataset.evidence import _event_for


LABELS = {
    "events": [
        {"soho_number": 5101, "group": "Kreutz", "source_line": "5101 ...",
         "notes": "", "days": {
             "2024-09-09": {"sequence_id": 36,
                            "positive_track_ids": ["seq36_x_00104"]},
             "2024-09-10": {"sequence_id": 37,
                            "positive_track_ids": ["seq37_x_00017"]}}},
        {"soho_number": 5100, "group": "Kreutz", "source_line": "5100 ...",
         "notes": "not recovered", "days": {
             "2024-09-11": {"sequence_id": 38, "positive_track_ids": []},
             "2024-09-13": {"sequence_id": None, "positive_track_ids": []}}},
    ],
    "negative_days": [],
}


def test_label_maps_positive_and_excluded(tmp_path):
    import json
    d = tmp_path / "dataset" / "v1"
    d.mkdir(parents=True)
    (d / "labels.json").write_text(json.dumps(LABELS), encoding="utf-8")
    comets, excluded = label_maps(tmp_path)
    assert comets == {"seq36_x_00104", "seq37_x_00017"}
    assert excluded == {38}  # ingested, no positives; None seq ignored


def test_event_for_confirmed_positive():
    ev = _event_for("seq36_x_00104", 36, LABELS)
    assert ev["soho_number"] == 5101
    assert ev["relation"] == "confirmed_positive"
    assert ev["day"] == "2024-09-09"


def test_event_for_unrecovered_day():
    ev = _event_for("seq38_x_00001", 38, LABELS)
    assert ev["soho_number"] == 5100
    assert ev["relation"] == "unrecovered_event_day"
    assert ev["notes"] == "not recovered"


def test_event_for_plain_background():
    assert _event_for("seq1_x_00001", 1, LABELS) is None
