"""Tests for ml.validation_summary pure logic (no DB, tmp data root)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.validation_summary import build_summary, to_markdown

LABELS = {
    "label_format": "labels_v1",
    "splits": {"train": [], "val": [], "test": ["2024-09-09", "2024-09-10"]},
    "negative_days": [],
    "events": [
        {"soho_number": 5101, "group": "Kreutz",
         "days": {"2024-09-09": {"sequence_id": 1,
                                 "positive_track_ids": ["seq1_aa_00001"]}}},
        {"soho_number": 5102, "group": "Kreutz",
         "days": {"2024-09-10": {"sequence_id": 2,
                                 "positive_track_ids": []}}},
        {"soho_number": 4999, "group": "Kreutz",
         "days": {"2020-01-01": {"sequence_id": 3,
                                 "positive_track_ids": ["seq3_aa_00001"]}}},
    ],
}

SCORES = {
    "seq1_aa_00001": 0.9,   # the recovered positive
    "seq1_aa_00002": 0.95,  # a higher-scoring negative in the same sequence
    "seq1_aa_00003": 0.1,
    "seq3_aa_00001": 0.5,   # positive of the unsplit event
}


def _data_root(tmp_path: Path) -> Path:
    seq_dir = tmp_path / "processed" / "candidate_crops" / "seq_1"
    seq_dir.mkdir(parents=True)
    (seq_dir / "crops_manifest.json").write_text(json.dumps({
        "track_files": ["seq1_aa_00001.npz", "seq1_aa_00002.npz",
                        "seq1_aa_00003.npz"],
    }))
    (seq_dir / "seq1_aa_00001.npz").write_bytes(b"x")
    return tmp_path


def test_build_summary(tmp_path):
    summary = build_summary(LABELS, SCORES, _data_root(tmp_path))
    assert summary["recovery"] == {"n_events": 3, "n_recovered": 2,
                                   "n_top10_in_sequence": 2}
    by_soho = {e["soho_number"]: e for e in summary["events"]}
    e = by_soho[5101]
    assert e["recovered"] and e["best_fusion"] == 0.9
    assert e["sequence_rank"] == 2 and e["sequence_pool"] == 3
    assert e["global_rank"] == 2 and e["global_pool"] == 4
    assert not by_soho[5102]["recovered"]
    assert by_soho[4999]["splits"] == ["unsplit"]
    # test split: 1 positive vs 2 negatives; positive ranks 2nd -> AP 0.5
    m = summary["split_metrics"]["test"]
    assert (m["n_positives"], m["n_negatives"]) == (1, 2)
    assert m["ap"] == 0.5
    assert summary["split_metrics"]["train"]["n_positives"] == 0


def test_markdown_render(tmp_path):
    md = to_markdown(build_summary(LABELS, SCORES, _data_root(tmp_path)))
    assert "2/3 labeled Sungrazer events recovered" in md
    assert "| 5101 | Kreutz | test | yes (1 track) | 0.9000 | 2/3 | 2/4 |" in md
    assert "| 5102 | Kreutz | test | no | — | — | — |" in md
    assert "| test | 1 | 2 |" in md


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
