"""FastAPI service over the candidates index (ship plan S3).

Run:  uvicorn api.main:app --reload  (from backend/, needs DATABASE_URL)

Read-only: every route is a query over the DB index plus, for evidence,
the config-hashed pipeline files. Nothing here writes.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from agent.discovery import run_agent
from dataset.evidence import build_evidence
from db.models import Candidate, ImageSequence, ModelPrediction
from db.session import SessionLocal

# Primary ranking signal (ship plan S2): mean of sigmoids across checkpoints.
FUSION_RUN_ID = "fusion_mean_v1"

app = FastAPI(title="SUNGRAZER AI", version="0.1.0")
# ponytail: allow-all CORS for local dev; restrict origins if ever deployed.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


def get_session():
    with SessionLocal() as session:
        yield session


def _data_root() -> Path:
    return Path(os.environ.get("SOHO_DATA_ROOT", "data"))


def _candidate_dict(c: Candidate, fusion_score: float | None) -> dict:
    return {
        "id": c.id,
        "track_id": c.track_id,
        "sequence_id": c.sequence_id,
        "n_frames": c.n_frames,
        "start_time": c.start_time.isoformat() if c.start_time else None,
        "end_time": c.end_time.isoformat() if c.end_time else None,
        "label": c.label,
        "status": c.status,
        "fusion_score": fusion_score,
    }


@app.get("/api/health")
def health(session: Session = Depends(get_session)) -> dict:
    session.execute(text("SELECT 1"))
    return {"status": "ok", "database": "ok"}


@app.get("/api/sequences")
def sequences(session: Session = Depends(get_session)) -> list[dict]:
    n_candidates = (
        select(Candidate.sequence_id, func.count().label("n"))
        .group_by(Candidate.sequence_id).subquery()
    )
    rows = session.execute(
        select(ImageSequence, n_candidates.c.n)
        .outerjoin(n_candidates, n_candidates.c.sequence_id == ImageSequence.id)
        .order_by(ImageSequence.id)
    ).all()
    return [{
        "id": s.id,
        "instrument": s.instrument,
        "start_time": s.start_time.isoformat() if s.start_time else None,
        "end_time": s.end_time.isoformat() if s.end_time else None,
        "frame_count": s.frame_count,
        "status": s.status,
        "n_candidates": n or 0,
    } for s, n in rows]


@app.get("/api/candidates")
def candidates(
    status: str | None = None,
    label: str | None = None,
    sequence_id: int | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
) -> dict:
    fusion = (
        select(ModelPrediction.candidate_id, ModelPrediction.score)
        .where(ModelPrediction.run_id == FUSION_RUN_ID).subquery()
    )
    query = select(Candidate, fusion.c.score).outerjoin(
        fusion, fusion.c.candidate_id == Candidate.id
    )
    if status is not None:
        query = query.where(Candidate.status == status)
    if label is not None:
        query = query.where(Candidate.label == label)
    if sequence_id is not None:
        query = query.where(Candidate.sequence_id == sequence_id)
    total = session.execute(
        select(func.count()).select_from(query.subquery())
    ).scalar_one()
    rows = session.execute(
        query.order_by(fusion.c.score.desc().nulls_last(), Candidate.id)
        .limit(limit).offset(offset)
    ).all()
    return {"total": total, "limit": limit, "offset": offset,
            "items": [_candidate_dict(c, s) for c, s in rows]}


@app.get("/api/candidates/{track_id}")
def candidate_detail(track_id: str,
                     session: Session = Depends(get_session)) -> dict:
    candidate = session.execute(
        select(Candidate).where(Candidate.track_id == track_id)
    ).scalar_one_or_none()
    if candidate is None:
        raise HTTPException(404, f"candidate {track_id!r} not found")
    predictions = session.execute(
        select(ModelPrediction)
        .where(ModelPrediction.candidate_id == candidate.id)
        .order_by(ModelPrediction.run_id)
    ).scalars().all()
    fusion = next((p.score for p in predictions if p.run_id == FUSION_RUN_ID),
                  None)
    out = _candidate_dict(candidate, fusion)
    out["features"] = candidate.features
    out["predictions"] = [{"run_id": p.run_id,
                           "model_version": p.model_version,
                           "score": p.score} for p in predictions]
    return out


@app.get("/api/candidates/{track_id}/evidence")
def candidate_evidence(track_id: str,
                       session: Session = Depends(get_session)) -> dict:
    try:
        return build_evidence(session, _data_root(), track_id)
    except KeyError:
        raise HTTPException(404, f"candidate {track_id!r} not found")
    except FileNotFoundError as exc:
        # In the DB but its pipeline files are missing: server-side data gap.
        raise HTTPException(500, f"evidence incomplete: {exc}")


@app.get("/api/candidates/{track_id}/report")
def candidate_report(track_id: str,
                     session: Session = Depends(get_session)) -> dict:
    try:
        return run_agent(session, _data_root(), track_id)
    except KeyError:
        raise HTTPException(404, f"candidate {track_id!r} not found")
    except FileNotFoundError as exc:
        raise HTTPException(500, f"evidence incomplete: {exc}")


@app.get("/api/statistics")
def statistics(session: Session = Depends(get_session)) -> dict:
    by_status = dict(session.execute(
        select(Candidate.status, func.count()).group_by(Candidate.status)
    ).all())
    by_label = dict(session.execute(
        select(Candidate.label, func.count())
        .where(Candidate.label.is_not(None)).group_by(Candidate.label)
    ).all())
    runs = [{"run_id": r, "model_version": v, "n_predictions": n}
            for r, v, n in session.execute(
                select(ModelPrediction.run_id, ModelPrediction.model_version,
                       func.count())
                .group_by(ModelPrediction.run_id,
                          ModelPrediction.model_version)
                .order_by(ModelPrediction.run_id)
            ).all()]
    return {
        "candidates_total": session.execute(
            select(func.count()).select_from(Candidate)).scalar_one(),
        "candidates_by_status": by_status,
        "candidates_by_label": by_label,
        "sequences_total": session.execute(
            select(func.count()).select_from(ImageSequence)).scalar_one(),
        "runs": runs,
    }
