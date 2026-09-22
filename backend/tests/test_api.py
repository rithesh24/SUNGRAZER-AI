"""Route tests for api.main against an in-memory SQLite database."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_, compiler, **kw):
    return "JSON"


from fastapi.testclient import TestClient

from api.main import app, get_session
from db.models import Base, Candidate, ImageSequence, Mission, ModelPrediction

engine = create_engine("sqlite://", poolclass=StaticPool,
                       connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)
TestSession = sessionmaker(bind=engine, expire_on_commit=False)
def _override_session():
    with TestSession() as s:
        yield s


app.dependency_overrides[get_session] = _override_session
client = TestClient(app)

T0 = datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2020, 1, 1, 6, 0, tzinfo=timezone.utc)
TRACKS = ["seq36_aaaa_00001", "seq36_aaaa_00002", "seq36_aaaa_00003"]


def _seed() -> None:
    with TestSession() as s:
        s.add(Mission(id=1, mission_name="SOHO", instrument_name="LASCO"))
        s.add(ImageSequence(id=36, mission_id=1, instrument="LASCO/C3",
                            start_time=T0, end_time=T1, frame_count=48,
                            status="PROCESSED"))
        rows = [
            Candidate(id=1, track_id=TRACKS[0], sequence_id=36, n_frames=9,
                      start_time=T0, end_time=T1, label="comet",
                      status="KNOWN_COMET", features={"speed": 1.5}),
            Candidate(id=2, track_id=TRACKS[1], sequence_id=36, n_frames=7,
                      status="HIGH_PRIORITY"),
            Candidate(id=3, track_id=TRACKS[2], sequence_id=36, n_frames=5,
                      status="LOW_PRIORITY"),
        ]
        s.add_all(rows)
        s.add_all([
            ModelPrediction(candidate_id=1, run_id="fusion_mean_v1",
                            model_version="temporal_ranker_v1", score=0.9),
            ModelPrediction(candidate_id=1, run_id="run_baseline_seed0",
                            model_version="temporal_ranker_v1", score=0.8),
            ModelPrediction(candidate_id=2, run_id="fusion_mean_v1",
                            model_version="temporal_ranker_v1", score=0.5),
        ])
        s.commit()


def _write_data_root() -> Path:
    root = Path(tempfile.mkdtemp(prefix="sungrazer_api_test_"))
    tracks_dir = root / "processed" / "tracks" / "seq_36"
    dna_dir = root / "processed" / "motion_dna" / "seq_36"
    labels_dir = root / "dataset" / "v1"
    for d in (tracks_dir, dna_dir, labels_dir):
        d.mkdir(parents=True)
    (tracks_dir / "tracks.json").write_text(json.dumps({
        "config_hash": "cfg123", "motion_config_hash": "mot456",
        "tracks": [{"track_id": t, "positions": [[1, 2]], "timestamps": [0]}
                   for t in TRACKS],
    }))
    (dna_dir / "motion_dna.json").write_text(json.dumps({
        "config_hash": "dna789", "sun_center_xy": [512, 512],
        "features": {t: {"speed": 1.5} for t in TRACKS},
    }))
    (labels_dir / "labels.json").write_text(json.dumps({
        "label_format": "labels_v1",
        "events": [{"soho_number": 5057, "group": "Kreutz",
                    "source_line": "SOHO-5057 line",
                    "days": {"2020-01-01": {
                        "sequence_id": 36,
                        "positive_track_ids": [TRACKS[0]]}}}],
    }))
    return root


_seed()
os.environ["SOHO_DATA_ROOT"] = str(_write_data_root())


def test_health():
    body = client.get("/api/health").json()
    assert body == {"status": "ok", "database": "ok"}


def test_sequences():
    body = client.get("/api/sequences").json()
    assert len(body) == 1
    assert body[0]["id"] == 36
    assert body[0]["instrument"] == "LASCO/C3"
    assert body[0]["n_candidates"] == 3


def test_candidates_ranked_by_fusion():
    body = client.get("/api/candidates").json()
    assert body["total"] == 3
    assert [i["track_id"] for i in body["items"]] == TRACKS  # 0.9, 0.5, None
    assert body["items"][0]["fusion_score"] == 0.9
    assert body["items"][2]["fusion_score"] is None


def test_candidates_filters_and_paging():
    body = client.get("/api/candidates", params={"status": "HIGH_PRIORITY"}).json()
    assert body["total"] == 1 and body["items"][0]["track_id"] == TRACKS[1]
    body = client.get("/api/candidates", params={"label": "comet"}).json()
    assert body["total"] == 1 and body["items"][0]["status"] == "KNOWN_COMET"
    body = client.get("/api/candidates", params={"limit": 1, "offset": 1}).json()
    assert body["total"] == 3 and body["items"][0]["track_id"] == TRACKS[1]


def test_candidate_detail():
    body = client.get(f"/api/candidates/{TRACKS[0]}").json()
    assert body["fusion_score"] == 0.9
    assert body["features"] == {"speed": 1.5}
    assert [p["run_id"] for p in body["predictions"]] == [
        "fusion_mean_v1", "run_baseline_seed0"]
    assert client.get("/api/candidates/seq99_nope_00000").status_code == 404


def test_evidence():
    body = client.get(f"/api/candidates/{TRACKS[0]}/evidence").json()
    assert body["track_id"] == TRACKS[0]
    assert body["candidate"]["status"] == "KNOWN_COMET"
    assert body["track"]["positions"] == [[1, 2]]
    assert body["motion_dna"] == {"speed": 1.5}
    assert body["event"]["soho_number"] == 5057
    assert body["event"]["relation"] == "confirmed_positive"
    assert body["provenance"]["tracks_config_hash"] == "cfg123"
    assert client.get("/api/candidates/seq99_nope_00000/evidence").status_code == 404


def test_evidence_missing_files_is_500():
    os.environ["SOHO_DATA_ROOT"], saved = "/nonexistent", os.environ["SOHO_DATA_ROOT"]
    try:
        # FileNotFoundError path: candidate in DB, pipeline files absent.
        response = client.get(f"/api/candidates/{TRACKS[0]}/evidence")
        assert response.status_code == 500
        assert "evidence incomplete" in response.json()["detail"]
    finally:
        os.environ["SOHO_DATA_ROOT"] = saved


def test_candidate_report_fallback():
    os.environ.pop("GROQ_API_KEY", None)  # force the deterministic path
    body = client.get(f"/api/candidates/{TRACKS[0]}/report").json()
    assert body["generated_by"] == "fallback"
    assert "## Assessment" in body["report"]
    assert body["event"]["soho_number"] == 5057
    assert client.get("/api/candidates/seq99_nope_00000/report").status_code == 404


def test_statistics():
    body = client.get("/api/statistics").json()
    assert body["candidates_total"] == 3
    assert body["candidates_by_status"] == {
        "KNOWN_COMET": 1, "HIGH_PRIORITY": 1, "LOW_PRIORITY": 1}
    assert body["candidates_by_label"] == {"comet": 1}
    assert body["sequences_total"] == 1
    assert {r["run_id"]: r["n_predictions"] for r in body["runs"]} == {
        "fusion_mean_v1": 2, "run_baseline_seed0": 1}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
