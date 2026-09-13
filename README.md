# SUNGRAZER AI

Autonomous scientific discovery of faint moving objects in SOHO/LASCO spacecraft imagery.

A local-first pipeline that ingests coronagraph image sequences, extracts persistent motion, ranks moving-object candidates with a temporal PyTorch model, filters known objects, and presents explainable candidates for human review.

## Documentation

- `docs/project.md` — full project definition and system flow
- `docs/techspec_groq.md` — technical specification
- `docs/tracker.md` — authoritative task checklist
- `docs/progress.md` — engineering journal / session handoffs
- `docs/claude.md` — development working rules

## Current status

Phase 1 (Data Foundation). Implemented so far:

- SOHO/LASCO ingestion prototype (`backend/ingestion/`) — downloads level-0.5 FITS files from the verified public archives with idempotent resume via per-day JSON manifests.
- PostgreSQL foundation (`backend/db/`, `backend/alembic/`) — dockerized PostgreSQL 17, SQLAlchemy models and Alembic migrations for `missions` / `image_sequences` / `images`, and an idempotent manifest→database backfill (`python -m ingestion.persist`).
- Sequence grouping (`backend/ingestion/sequences.py`) — assigns ingested images to chronological, per-instrument `image_sequences`, splitting on time gaps (default 60 min), with integrity checks on every run.
- FITS handling (`backend/pipeline/fits_io.py`) — validated loader producing a `ScienceImage` model (UTC timestamp, detector, exposure, sun-center WCS, full original header) with native-byte-order pixel data ready for OpenCV.
- Preprocessing baseline (`backend/pipeline/preprocess.py`) — exposure-normalized float32 frames with occulter/FOV/border masks and QA, written as `.npz` under `data/processed/normalized/seq_<id>/` with a versioned config manifest (`python -m pipeline.preprocess --sequence-id N`).
- Registration baseline (`backend/pipeline/register.py`) — phase-correlation translation alignment to a sequence reference frame with a quality metric and explicit failure recording; outputs under `data/processed/registered/seq_<id>/` (`python -m pipeline.register --sequence-id N`, `--evaluate` for the method-comparison experiment).
- Temporal motion extraction (`backend/pipeline/motion.py`) — temporal-median background subtraction, robust-sigma thresholding, connected-component detections per frame; outputs `data/processed/motion/seq_<id>/detections.json` (`python -m pipeline.motion --sequence-id N`, `--evaluate` for the synthetic-recovery method comparison).
- Candidate tracking (`backend/pipeline/tracks.py`) — greedy nearest-neighbor association of detections into multi-frame tracks with velocity prediction, gap coasting, and persistence filtering; outputs `data/processed/tracks/seq_<id>/tracks.json` (`python -m pipeline.tracks --sequence-id N`).

## Ingestion prototype — setup and usage

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Download (up to) 5 files of LASCO C3 for one day
python -m ingestion.ingest --date 2024-01-01 --camera c3 --limit 5

# Offline unit tests
python tests/test_ingestion.py
```

Data lands under `data/raw/soho/lasco/<YYMMDD>/<camera>/`, alongside a `manifest.json` recording checksum, size, source URL, and FITS `DATE-OBS` for each file. Re-running the same command skips already-ingested files.

Configuration is environment-driven — see `.env.example`.

## Database — setup and usage

Requires Docker Desktop.

```bash
# From the repository root: start PostgreSQL 17 (data persists in a named volume)
docker compose up -d

cd backend
.venv\Scripts\activate        # Windows

# Apply migrations (URL comes from DATABASE_URL, defaults to the compose service)
alembic upgrade head

# Persist downloaded-image metadata from the day manifests into the images table
python -m ingestion.persist

# Group persisted images into chronological per-camera sequences
python -m ingestion.sequences            # add --check-only to only validate

# Offline unit tests
python tests/test_persist.py
python tests/test_sequences.py
```

Re-running `ingestion.persist` is idempotent: unchanged manifest entries are skipped, changed ones (e.g. a failed download that later succeeded) are updated in place.

## Data sources

- Historical (full mission): `https://lasco-www.nrl.navy.mil/lz/level_05/` (NRL)
- Recent (~2 weeks): `https://umbra.nascom.nasa.gov/pub/lasco/lastimage/level_05/` (NASA SDAC)

Both verified 2026-09-12; layout `<base>/<YYMMDD>/<c2|c3>/<8-digit-id>.fts`.
