"""Tests for agent.discovery: compact prompt input, fallback report, and
the failure-safe degrade path. No DB, no network."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import discovery

EVIDENCE = {
    "track_id": "seq36_aaaa_00001",
    "candidate": {"sequence_id": 36, "instrument": "LASCO/C3",
                  "start_time": "2020-01-01T00:00:00+00:00",
                  "end_time": "2020-01-01T06:00:00+00:00",
                  "n_frames": 9, "label": "comet", "status": "KNOWN_COMET"},
    "track": {"track_id": "seq36_aaaa_00001",
              "positions": [[1, 2], [3, 4], [5, 6]],
              "timestamps": [0, 1, 2]},
    "motion_dna": {"speed_px_frame": 2.83, "sunward_cos": 0.97,
                   "flags": ["ok"]},
    "predictions": [
        {"run_id": "fusion_mean_v1", "model_version": "temporal_ranker_v1",
         "score": 0.9},
        {"run_id": "run_baseline_seed0",
         "model_version": "temporal_ranker_v1", "score": 0.8},
    ],
    "event": {"soho_number": 5057, "relation": "confirmed_positive",
              "day": "2020-01-01", "group": "Kreutz",
              "source_line": "x", "notes": None},
    "provenance": {"tracks_config_hash": "cfg123"},
    "caveats": ["Scores are ranking signals.", "Not a discovery."],
}


def test_compact_drops_per_frame_arrays():
    compact = discovery._compact(EVIDENCE)
    assert "track" not in compact and "provenance" not in compact
    assert compact["track_summary"] == {"n_points": 3,
                                        "first_position": [1, 2],
                                        "last_position": [5, 6]}
    assert compact["motion_dna"] == EVIDENCE["motion_dna"]
    assert compact["event"]["soho_number"] == 5057


def test_fallback_report_quotes_evidence():
    report = discovery._fallback_report(EVIDENCE)
    assert "## Assessment" in report and "## Key evidence" in report
    assert "## Caveats" in report
    assert "0.9000" in report          # fusion score
    assert "run_baseline_seed0" in report
    assert "speed_px_frame" in report and "2.83" in report
    assert "SOHO-5057" in report and "confirmed_positive" in report
    assert "Scores are ranking signals." in report
    assert "flags" not in report       # non-numeric DNA entries skipped


def test_fallback_without_event_or_predictions():
    evidence = {**EVIDENCE, "event": None, "predictions": [],
                "motion_dna": None}
    report = discovery._fallback_report(evidence)
    assert "No known-event attribution." in report


def test_run_agent_degrades_when_llm_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(discovery, "build_evidence",
                        lambda session, root, tid: EVIDENCE)
    monkeypatch.setattr(discovery, "_call_llm",
                        lambda evidence: (_ for _ in ()).throw(
                            RuntimeError("groq down")))
    result = discovery.run_agent(None, tmp_path, "seq36_aaaa_00001")
    assert result["generated_by"] == "fallback"
    assert "## Assessment" in result["report"]
    assert result["event"]["soho_number"] == 5057


def test_run_agent_uses_llm_when_available(monkeypatch, tmp_path):
    monkeypatch.setattr(discovery, "build_evidence",
                        lambda session, root, tid: EVIDENCE)
    monkeypatch.setattr(discovery, "_call_llm",
                        lambda evidence: "## Assessment\nLLM text")
    result = discovery.run_agent(None, tmp_path, "seq36_aaaa_00001")
    assert result["generated_by"].startswith("groq:")
    assert result["report"] == "## Assessment\nLLM text"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
