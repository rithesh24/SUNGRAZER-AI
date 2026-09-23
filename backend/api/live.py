"""Live-analysis endpoints: pull a recent day from NASA's rolling archive and
run the full pipeline on it as a single background job with polled status.

Each stage runs as a subprocess of the documented ``python -m`` CLI chain
(ingest -> persist -> sequences -> preprocess -> register -> motion -> tracks
-> motion_dna -> crops -> candidates -> score), so behavior is identical to
running the stages by hand and every idempotency/config-hash guarantee holds.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select

from db.models import ImageSequence
from db.session import SessionLocal
from ingestion import archive_client, config

router = APIRouter()

INSTRUMENT = "LASCO/C3"
CAMERA = "c3"
RECENT_WINDOW_DAYS = 14
AVAILABLE_CACHE_SECONDS = 600
LOG_TAIL_LINES = 40
# Checkpoint pair behind fusion_mean_v1 (see ml.score docstring).
SCORE_RUNS = (
    "run_f7ded5030483e421_dataa3de09e1_seed0",
    "run_744ca82263542eda_dataaa66d7b4_seed0",
)
BACKEND_DIR = Path(__file__).resolve().parents[1]

# ponytail: single in-memory job, one at a time; a job table if this ever
# needs multiple workers or survival across API restarts.
_lock = threading.Lock()
_job: dict = {
    "state": "idle", "date": None, "stage": None,
    "stages_done": 0, "stages_total": 0,
    "log_tail": [], "sequence_ids": [], "error": None,
}
_avail_cache: tuple[float, list[dict]] | None = None


def _set(**kwargs) -> None:
    with _lock:
        _job.update(kwargs)


def _append_log(line: str) -> None:
    with _lock:
        _job["log_tail"] = (_job["log_tail"] + [line])[-LOG_TAIL_LINES:]


def _sequence_ids_for(day: date) -> list[int]:
    with SessionLocal() as session:
        rows = session.execute(
            select(ImageSequence.id)
            .where(
                ImageSequence.instrument == INSTRUMENT,
                func.date(ImageSequence.start_time) == day,
            )
            .order_by(ImageSequence.id)
        ).scalars().all()
    return list(rows)


def _run_stage(name: str, argv: list[str]) -> None:
    _set(stage=name)
    _append_log(f"── {name}: python -m {' '.join(argv)}")
    # Fail fast on SDAC's flaky connections: a stalled request costs 20s
    # before the (reliably successful) retry, not the default 60s.
    env = {"LASCO_HTTP_TIMEOUT_SECONDS": "20", **os.environ}
    proc = subprocess.Popen(
        [sys.executable, "-m", *argv],
        cwd=BACKEND_DIR, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.strip():
            _append_log(line.rstrip())
    if proc.wait() != 0:
        raise RuntimeError(f"stage '{name}' failed (exit {proc.returncode}) — "
                           "see the log above")
    with _lock:
        _job["stages_done"] += 1


def _run_job(day: date) -> None:
    iso = day.isoformat()
    per_seq_modules = (
        "pipeline.preprocess", "pipeline.register", "pipeline.motion",
        "pipeline.tracks", "pipeline.motion_dna", "ml.crops",
    )
    try:
        _run_stage("download imagery", ["ingestion.ingest", "--date", iso,
                                        "--camera", CAMERA])
        _run_stage("persist images", ["ingestion.persist"])
        _run_stage("build sequences", ["ingestion.sequences"])
        seq_ids = _sequence_ids_for(day)
        if not seq_ids:
            raise RuntimeError(
                "no frames for this day made it into a sequence — the archive "
                "may not have data for it yet")
        _set(sequence_ids=seq_ids,
             stages_total=3 + len(seq_ids) * len(per_seq_modules) + 2)
        for seq in seq_ids:
            for mod in per_seq_modules:
                _run_stage(f"{mod.rsplit('.', 1)[-1]} · seq {seq}",
                           [mod, "--sequence-id", str(seq)])
        seq_args: list[str] = []
        for seq in seq_ids:
            seq_args += ["--sequence-id", str(seq)]
        _run_stage("persist candidates", ["dataset.candidates", *seq_args])
        run_args: list[str] = []
        for run in SCORE_RUNS:
            run_args += ["--run", run]
        _run_stage("model scoring", ["ml.score", *seq_args, *run_args])
        _set(state="done", stage=None)
    except Exception as exc:  # surfaced via /status, never crashes the API
        _set(state="error", error=str(exc))


@router.get("/api/live/available")
def live_available() -> dict:
    global _avail_cache
    now = time.monotonic()
    if _avail_cache and now - _avail_cache[0] < AVAILABLE_CACHE_SECONDS:
        days = _avail_cache[1]
    else:
        today = datetime.now(timezone.utc).date()
        dates = [today - timedelta(days=i) for i in range(RECENT_WINDOW_DAYS)]

        def frames_on_archive(d: date) -> int:
            try:
                return len(archive_client.list_day_files(
                    d, CAMERA, config.RECENT_BASE_URL))
            except Exception:
                return 0

        with ThreadPoolExecutor(max_workers=6) as pool:
            counts = list(pool.map(frames_on_archive, dates))
        days = [{"date": d.isoformat(), "frames": n}
                for d, n in zip(dates, counts)]
        _avail_cache = (now, days)
    return {"days": [
        {**d, "ingested": bool(ids), "sequence_ids": ids}
        for d in days
        for ids in [_sequence_ids_for(date.fromisoformat(d["date"]))]
    ]}


@router.get("/api/live/status")
def live_status() -> dict:
    with _lock:
        return dict(_job)


class RunBody(BaseModel):
    date: str


@router.post("/api/live/run")
def live_run(body: RunBody) -> dict:
    try:
        day = date.fromisoformat(body.date)
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid date")
    with _lock:
        if _job["state"] == "running":
            raise HTTPException(status_code=409,
                                detail="a live analysis is already running")
        _job.update(state="running", date=day.isoformat(), stage="starting",
                    stages_done=0, stages_total=11, log_tail=[],
                    sequence_ids=[], error=None)
    threading.Thread(target=_run_job, args=(day,), daemon=True).start()
    with _lock:
        return dict(_job)
