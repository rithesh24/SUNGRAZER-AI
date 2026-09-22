# SUNGRAZER AI

Autonomous detection and prioritization of faint sungrazing-comet candidates in SOHO/LASCO coronagraph imagery — a local-first scientific pipeline with a mission-control web interface.

**Validated result:** run on 41 day-sequences of 2024 LASCO C3 archive data, the pipeline extracted 34,994 moving-object tracks and recovered **10 of 15 confirmed Sungrazer events — every one ranked #1 inside its own day-sequence** by the model's fusion score. Held-out test AP 0.4613 (7 positives vs 7,484 negatives). Full numbers and caveats: [`docs/validation.md`](docs/validation.md).

## How it works

```
SOHO/LASCO archive (FITS, level 0.5)
  → ingestion (idempotent, manifest-tracked)          backend/ingestion/
  → preprocessing (normalize, occulter/FOV masks)     backend/pipeline/preprocess.py
  → registration (phase-correlation alignment)        backend/pipeline/register.py
  → motion extraction (temporal-median subtraction)   backend/pipeline/motion.py
  → tracking (multi-frame association, coasting)      backend/pipeline/tracks.py
  → Motion DNA (behavioral features per track)        backend/pipeline/motion_dna.py
  → PyTorch temporal ranker (CNN+GRU on crop stacks)  backend/ml/
  → PostgreSQL index + evidence builder               backend/db/, backend/dataset/
  → FastAPI                                           backend/api/
  → discovery agent (LangGraph + Groq, failure-safe)  backend/agent/
  → React mission-control UI                          frontend/
```

The scientific core is deterministic and provenance-hashed at every stage; the LLM only narrates structured evidence, and degrades to a deterministic report when unavailable. Candidates are ranking signals for human review — never claimed discoveries.

## The interface

Six views on an animated starfield: **Overview** (mission statistics), **Candidates** (ranked explorer with filters and track search), **Candidate detail** (animated detection crops, trajectory with Sun sight-line, Motion DNA, per-run scores, known-object attribution, provenance, agent report, human review panel), **Review queue** (high-priority tracks, best first), **Archaeology** (Motion-DNA similarity search across the whole archive — querying a known comet returns the other known comets), and **Unknown objects**.

## Running it

Requires: Docker Desktop, Python 3.13, Node 22.

```bash
# 1. PostgreSQL
docker compose up -d

# 2. Backend (from backend/)
python -m venv .venv
.venv\Scripts\activate                  # Windows
pip install -r requirements.txt
alembic upgrade head
uvicorn api.main:app --port 8000

# 3. Frontend (from frontend/)
npm install
npm run dev                             # http://localhost:5173 (proxies /api → :8000)
```

Configuration is environment-driven — copy `.env.example` to `.env` at the repo root (loaded automatically). `GROQ_API_KEY` is optional: without it, agent reports use the deterministic fallback.

To process data from scratch (per sequence): `ingestion.ingest` → `ingestion.persist` → `ingestion.sequences` → `pipeline.preprocess` → `pipeline.register` → `pipeline.motion` → `pipeline.tracks` → `pipeline.motion_dna` → `ml.crops` → `ml.score`. Every stage is an idempotent `python -m` CLI writing config-hashed outputs under `data/processed/`.

```bash
# Tests (from backend/)
python -m pytest tests -q               # 100+ offline tests, no network/GPU needed

# Reproduce the validation summary
python -m ml.validation_summary
```

## Documentation

- [`docs/validation.md`](docs/validation.md) — scientific validation: recovery table, ranking metrics, limitations
- [`docs/project.md`](docs/project.md) — full project definition and system flow
- [`docs/techspec_groq.md`](docs/techspec_groq.md) — technical specification
- [`docs/tracker.md`](docs/tracker.md) — authoritative task checklist (including what was descoped and why)
- [`docs/progress.md`](docs/progress.md) — engineering journal, session by session
- [`docs/claude.md`](docs/claude.md) — development working rules

## Data sources

- Historical (full mission): `https://lasco-www.nrl.navy.mil/lz/level_05/` (NRL)
- Recent (~2 weeks): `https://umbra.nascom.nasa.gov/pub/lasco/lastimage/level_05/` (NASA SDAC)
- Ground truth: the [Sungrazer Project](https://sungrazer.nrl.navy.mil/) confirmed-comet lists

Both archives verified 2026-09-12; layout `<base>/<YYMMDD>/<c2|c3>/<8-digit-id>.fts`.

## Honest limitations

Batch mode only (no live polling — by design for a local-first tool); LASCO C3 / 2024 / Kreutz-group coverage; 22 positive tracks is a small validation set; scores are uncalibrated ranking signals; confirmation of any candidate requires human review and astrometric verification. Details in [`docs/validation.md`](docs/validation.md).
