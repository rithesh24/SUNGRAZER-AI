"""FastAPI service over the candidates index (ship plan S3).

Run:  uvicorn api.main:app --reload  (from backend/, needs DATABASE_URL)

Read-only: every route is a query over the DB index plus, for evidence,
the config-hashed pipeline files. Nothing here writes.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from pydantic import BaseModel, field_validator

from agent.discovery import run_agent
from dataset.evidence import build_evidence
from db.models import REVIEW_VERDICTS, Candidate, ImageSequence, ModelPrediction
from db.session import SessionLocal

# Primary ranking signal (ship plan S2): mean of sigmoids across checkpoints.
FUSION_RUN_ID = "fusion_mean_v1"

from api.live import router as live_router

app = FastAPI(title="SUNGRAZER AI", version="0.1.0")
# ponytail: allow-all CORS for local dev; restrict origins if ever deployed.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])
app.include_router(live_router)


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
        "review": c.review,
        "reviewer_notes": c.reviewer_notes,
        "reviewed_at": c.reviewed_at.isoformat() if c.reviewed_at else None,
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
    q: str | None = None,
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
    if q:
        query = query.where(Candidate.track_id.ilike(f"%{q}%"))
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


# Motion-DNA matrix for similarity search, built once per process.
# ponytail: module-level cache; dataset is frozen — restart to refresh.
_DNA_CACHE: dict | None = None


def _dna_cache(session: Session) -> dict:
    global _DNA_CACHE
    if _DNA_CACHE is None:
        rows = session.execute(
            select(Candidate.id, Candidate.track_id, Candidate.features)
            .where(Candidate.features.is_not(None))
        ).all()
        keys = sorted({k for _, _, f in rows for k, v in f.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)})
        matrix = np.array([[float(f.get(k, 0.0)) if isinstance(f.get(k), (int, float)) else 0.0
                            for k in keys] for _, _, f in rows], dtype=np.float64)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std[std == 0] = 1.0
        _DNA_CACHE = {
            "track_ids": [t for _, t, _ in rows],
            "index": {t: i for i, (_, t, _) in enumerate(rows)},
            "z": (matrix - mean) / std,
        }
    return _DNA_CACHE


@app.get("/api/candidates/{track_id}/similar")
def candidate_similar(track_id: str, limit: int = Query(20, ge=1, le=100),
                      session: Session = Depends(get_session)) -> dict:
    """Nearest tracks by Motion DNA (z-scored L2 distance, lower = closer).

    The honest 'trajectory archaeology': which tracks anywhere in the
    processed archive move like this one.
    """
    cache = _dna_cache(session)
    position = cache["index"].get(track_id)
    if position is None:
        raise HTTPException(404, f"candidate {track_id!r} has no Motion DNA")
    distances = np.linalg.norm(cache["z"] - cache["z"][position], axis=1)
    order = np.argsort(distances)
    nearest = [i for i in order if i != position][:limit]
    ids = [cache["track_ids"][i] for i in nearest]
    fusion = (
        select(ModelPrediction.candidate_id, ModelPrediction.score)
        .where(ModelPrediction.run_id == FUSION_RUN_ID).subquery()
    )
    rows = session.execute(
        select(Candidate, fusion.c.score)
        .outerjoin(fusion, fusion.c.candidate_id == Candidate.id)
        .where(Candidate.track_id.in_(ids))
    ).all()
    by_track = {c.track_id: (c, s) for c, s in rows}
    matches = []
    for i in nearest:
        tid = cache["track_ids"][i]
        if tid not in by_track:
            continue
        c, score = by_track[tid]
        item = _candidate_dict(c, score)
        item["dna_distance"] = round(float(distances[i]), 4)
        matches.append(item)
    return {"track_id": track_id, "matches": matches}


class ReviewBody(BaseModel):
    """Human review write: verdict and/or notes. Trust-boundary validated."""

    review: str | None = None
    reviewer_notes: str | None = None

    @field_validator("review")
    @classmethod
    def _known_verdict(cls, value: str | None) -> str | None:
        if value is not None and value not in REVIEW_VERDICTS:
            raise ValueError(f"review must be one of {REVIEW_VERDICTS}")
        return value

    @field_validator("reviewer_notes")
    @classmethod
    def _notes_length(cls, value: str | None) -> str | None:
        if value is not None and len(value) > 10_000:
            raise ValueError("notes too long (max 10000 chars)")
        return value


@app.patch("/api/candidates/{track_id}/review")
def candidate_review(track_id: str, body: ReviewBody,
                     session: Session = Depends(get_session)) -> dict:
    """Set or clear the human verdict/notes. Never touches ``status``."""
    candidate = session.execute(
        select(Candidate).where(Candidate.track_id == track_id)
    ).scalar_one_or_none()
    if candidate is None:
        raise HTTPException(404, f"candidate {track_id!r} not found")
    fields = body.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(422, "nothing to update")
    for key, value in fields.items():
        setattr(candidate, key, value)
    candidate.reviewed_at = (
        func.now() if (candidate.review or candidate.reviewer_notes) else None
    )
    session.commit()
    session.refresh(candidate)
    fusion = session.execute(
        select(ModelPrediction.score).where(
            ModelPrediction.candidate_id == candidate.id,
            ModelPrediction.run_id == FUSION_RUN_ID)
    ).scalar_one_or_none()
    return _candidate_dict(candidate, fusion)


@app.get("/api/candidates/{track_id}/crops.png")
def candidate_crops(track_id: str, scale: int = Query(4, ge=1, le=8),
                    session: Session = Depends(get_session)) -> Response:
    """Crop stack rendered as a horizontal PNG filmstrip.

    Frame count = image width / (32 * scale); the frontend animates it as
    a flipbook. Contrast is a robust percentile stretch over the stack.
    """
    candidate = session.execute(
        select(Candidate).where(Candidate.track_id == track_id)
    ).scalar_one_or_none()
    if candidate is None:
        raise HTTPException(404, f"candidate {track_id!r} not found")
    path = (_data_root() / "processed" / "candidate_crops"
            / f"seq_{candidate.sequence_id}" / f"{track_id}.npz")
    if not path.exists():
        raise HTTPException(404, f"no crop stack for {track_id!r}")
    crops = np.load(path)["crops"]  # float32 [T, K, K]
    lo, hi = np.percentile(crops, [1.0, 99.5])
    stack = np.clip((crops - lo) / max(hi - lo, 1e-6), 0, 1)
    strip = (np.hstack(list(stack)) * 255).astype(np.uint8)
    strip = cv2.resize(strip, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_NEAREST)
    ok, png = cv2.imencode(".png", strip)
    if not ok:
        raise HTTPException(500, "PNG encoding failed")
    return Response(content=png.tobytes(), media_type="image/png",
                    headers={"Cache-Control": "max-age=3600"})


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
