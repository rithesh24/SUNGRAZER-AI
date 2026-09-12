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

Early Phase 1 (Data Foundation). Implemented so far: the SOHO/LASCO ingestion prototype (`backend/ingestion/`) — downloads level-0.5 FITS files from the verified public archives with idempotent resume via per-day JSON manifests.

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

## Data sources

- Historical (full mission): `https://lasco-www.nrl.navy.mil/lz/level_05/` (NRL)
- Recent (~2 weeks): `https://umbra.nascom.nasa.gov/pub/lasco/lastimage/level_05/` (NASA SDAC)

Both verified 2026-09-12; layout `<base>/<YYMMDD>/<c2|c3>/<8-digit-id>.fts`.
