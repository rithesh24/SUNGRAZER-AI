# SUNGRAZER AI

Autonomous detection and prioritization of faint sungrazing-comet candidates in SOHO/LASCO coronagraph imagery — a local-first scientific pipeline with a mission-control web interface.

The system transforms raw coronagraph FITS frames from NASA/NRL archives into ranked, evidence-backed comet candidates using a deterministic computer-vision pipeline, a PyTorch temporal ranking model, and an LLM discovery agent that narrates — but never invents — the evidence.

Unlike typical AI applications, this system is designed as a **scientific triage engine**, combining:

- Deterministic, provenance-hashed image processing (every stage reproducible)
- Multi-frame motion tracking with physical plausibility checks
- Learned temporal ranking (CNN + GRU over detection crop stacks)
- Behavioral "Motion DNA" features enabling archive-wide similarity search
- Failure-safe LLM narration (Groq + LangGraph, deterministic fallback)
- A human review workflow — candidates are ranking signals, never claimed discoveries

**Validated result:** run on 41 day-sequences of 2024 LASCO C3 archive data, the pipeline extracted 34,994 moving-object tracks and recovered **10 of 15 confirmed Sungrazer events — every one ranked #1 inside its own day-sequence** by the model's fusion score. Held-out test AP 0.4613 (7 positives vs 7,484 negatives). 

---

## Core Idea

The SOHO spacecraft's LASCO coronagraphs have incidentally become the most prolific comet-discovery instrument in history , thousands of sungrazing comets have been found in its imagery, almost all by volunteers visually scanning frames. Most are members of the Kreutz group: faint, fast-moving smudges a few pixels across, diving toward the Sun.

SUNGRAZER AI automates the triage: it ingests a day of coronagraph imagery, finds everything that moves, characterizes *how* it moves, and ranks each track by how comet-like it is — so a human reviewer looks at the top handful of candidates instead of thousands of frames.

The design principle throughout: **the scientific core is deterministic and auditable; learning and language sit on top as ranking and narration layers.**

---

## Key Capabilities

### End-to-End Automated Triage
One command chain (or one click on the Live page) takes a calendar day from "raw FITS on a NASA server" to "ranked candidates with evidence packages in a database."

### Multi-Frame Motion Tracking
Detections are associated across frames with velocity prediction and coasting through missed frames, producing tracks with full position/time/photometry history.

### Motion DNA
Every track gets a behavioral fingerprint — speed, direction consistency, acceleration, curvature, radial (sunward) motion, flux evolution, frame coverage — used both as model features and for archive-wide similarity search ("trajectory archaeology").

### Learned Temporal Ranking
A deliberately small CNN+GRU (~60k parameters, sized for a few dozen positive examples) scores each track's crop stack. Two independently trained checkpoints are fused for the primary ranking signal.

### Known-Object Attribution
Candidates are cross-referenced against the Sungrazer Project's confirmed-comet lists, so known events are labeled as validation cases rather than surfaced as "finds."

### Evidence, Not Verdicts
Every candidate exposes a full evidence package: animated detection crops, trajectory plot with Sun sight-line, Motion DNA features, per-checkpoint model scores, provenance hashes, and attribution. The LLM agent writes a structured report *from* this package and is prompt-forbidden from claiming discoveries or inventing numbers.

### Human Review Workflow
Reviewers record verdicts and notes per candidate via the API/UI; review state is tracked separately from pipeline status and never overwritten by re-scoring.

---

## System Architecture

```
        SOHO/LASCO archives (FITS, level 0.5)
        NRL (full mission)  ·  NASA SDAC (rolling ~2 weeks)
                          │
        ┌─────────────────▼──────────────────┐
        │  Ingestion  (backend/ingestion/)    │
        │  parallel downloads · manifest      │
        │  tracking · FITS validation         │
        └─────────────────┬──────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │  Pipeline  (backend/pipeline/)      │
        │  preprocess → register → motion     │
        │  → tracks → motion_dna              │
        │  (config-hashed outputs per stage)  │
        └─────────────────┬──────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │  ML  (backend/ml/)                  │
        │  crop extraction → CNN+GRU ranker   │
        │  → checkpoint fusion → priority     │
        └─────────────────┬──────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │  PostgreSQL index  (backend/db/)    │
        │  sequences · images · candidates    │
        │  · predictions · reviews            │
        └─────────────────┬──────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │  FastAPI  (backend/api/)            │
        │  candidates · evidence · similarity │
        │  · review · live analysis · agent   │
        └───────┬─────────────────┬──────────┘
                │                 │
   ┌────────────▼─────────┐  ┌────▼─────────────────────┐
   │ Discovery agent      │  │ React mission-control UI │
   │ (LangGraph + Groq,   │  │ 6 views on an animated   │
   │  failure-safe)       │  │ starfield  (frontend/)   │
   └──────────────────────┘  └──────────────────────────┘
```

---

## The Processing Pipeline

Every stage is an idempotent `python -m` CLI that writes config-hashed outputs under `data/processed/` — running a stage twice with the same inputs and config is a no-op, and changing a config produces a new hash rather than overwriting results.

### Stage 1 — Ingestion (`ingestion.ingest`)
Downloads one day/camera of level-0.5 FITS files. Tries the NRL historical archive first, falls back to NASA SDAC's rolling recent archive. Key properties:

- **Parallel downloads** (default 6 workers, `LASCO_DOWNLOAD_WORKERS`) — SDAC stalls the first request for each file while it warms the cache, so serial ingest paid ~60 s/frame; overlapping the stalls recovers most of it.
- **Atomic writes** — each file downloads to `<name>.part` and renames on success, so an interrupted download never leaves a truncated file that looks complete.
- **Manifest tracking** — a per-day JSON manifest records status, SHA-256, size, and DATE-OBS per file. Completed files are skipped on re-run; failed files are retried.
- **FITS validation** — every stored file must load as readable FITS with 2-D data and a parseable observation time before being marked complete.
- **Partial-failure tolerance** — a few undownloadable frames are logged and tolerated; the stage fails only if *nothing* was ingested.

### Stage 2 — Persistence (`ingestion.persist`)
Registers ingested files into the PostgreSQL `images` table (idempotent upsert keyed on content).

### Stage 3 — Sequence Building (`ingestion.sequences`)
Groups a day's frames into `image_sequences` by observation time, splitting when the gap between consecutive frames exceeds `SEQUENCE_GAP_MINUTES` (default 60 — tolerates a few missed frames at LASCO C3's ~12-minute cadence without merging across real observation breaks). Integrity checks run over every sequence.

### Stage 4 — Preprocessing (`pipeline.preprocess`)
Per frame: normalization, occulter and field-of-view masking, and quality assessment (exposure check, Sun-center presence, invalid-pixel fraction). Frames failing QA are flagged unusable but recorded.

### Stage 5 — Registration (`pipeline.register`)
Aligns all frames of a sequence to a reference frame via phase correlation (Hanning-windowed, with an ECC refinement path). Frames with mismatched dimensions (SDAC occasionally serves 512×512 instead of 1024×1024) are excluded with a warning rather than corrupting the stack.

### Stage 6 — Motion Extraction (`pipeline.motion`)
Temporal-median subtraction across the registered stack isolates moving objects from the (static) corona and starfield; sigma-thresholded blob detection produces per-frame detections with position, area, peak, flux, and SNR (capped per frame to bound downstream cost).

### Stage 7 — Tracking (`pipeline.tracks`)
Associates detections across frames into tracks:

- Constant-velocity prediction per open track; greedy one-to-one assignment by distance within a gate radius (`max_link_px` = 10).
- Tracks **coast** through up to 2 missed frames before terminating.
- Persistence filter: only tracks spanning ≥ 5 frames are kept.
- **Duplicate-timestamp defense**: the recent archive sometimes re-serves a frame under a second file id with an identical DATE-OBS; zero-dt observations would break velocity math, so only the first frame per timestamp is tracked (plus a zero-span guard in the predictor).
- Track IDs are deterministic within (sequence, config): `seq<id>_<confighash>_<index>`.

### Stage 8 — Motion DNA (`pipeline.motion_dna`)
Computes the behavioral feature vector per track (see below), with range/finiteness validation on every feature — violations are logged per track and counted.

### Stage 9 — Crop Extraction (`ml.crops`)
Cuts a 32×32 pixel crop around each detection, producing a `[T, 32, 32]` stack per track (stored as `.npz`) — the model's visual input.

### Stage 10 — Candidate Persistence (`dataset.candidates`)
Upserts tracks + features into the `candidates` table, preserving any existing labels and review state.

### Stage 11 — Scoring (`ml.score`)
Runs the trained checkpoints over all crop stacks, writes per-run scores and the fusion score, and assigns priority status (see below).

---

## Motion DNA

Each track's behavioral fingerprint, computed from positions, timestamps, and photometry:

| Feature | Meaning |
|---|---|
| `n_frames`, `frame_coverage` | persistence; fraction of sequence epochs covered between first and last detection (computed on **timestamps**, not file order — archive file ids are not reliably chronological) |
| `duration_s`, `path_length_px`, `displacement_px` | extent of motion |
| `speed_mean_px_s`, `speed_std_px_s` | how fast, how steadily |
| `direction_deg`, `direction_consistency` | where it's heading and how straight |
| `accel_mean_px_s2`, `curvature_rad_px`, `linear_rms_px` | higher-order motion (null for tracks too short to estimate) |
| `flux_mean`, `flux_cv`, `flux_slope_frac_h` | brightness and brightening/fading trend |
| `r_min_px`, `r_max_px`, `radial_speed_px_s` | sunward geometry — Kreutz comets move *toward* the Sun |

Design choices worth noting: undefined values are `None`, never fabricated (a static track has no direction); every feature is range-validated; and the same vector doubles as the **similarity-search** basis — z-scored L2 distance over the full archive answers "which tracks anywhere move like this one."

---

## Machine Learning

### Model — `temporal_ranker_v1` (`ml/net.py`)

The smallest architecture that sees both appearance and temporal behavior:

- **CropEncoder**: 3 conv blocks + global average pool → 64-d embedding per 32×32 crop.
- **GRU** over the frame-embedding sequence (packed sequences mask zero-padded tails; true lengths are respected).
- Optional **Motion DNA concatenation** to the GRU hidden state before the head (ablation experiment; crop-only is the baseline).
- Linear scoring head → one logit per track (sigmoid → ranking score in [0, 1]).

**~60k parameters, deliberately** — with single-digit-to-low-double-digit positive examples, anything larger memorizes.

### Training (`ml/train.py`, `ml/dataset.py`, `ml/metrics.py`)

- Positives: tracks attributed to confirmed Sungrazer events; negatives: everything else.
- Evaluation: Average Precision on a held-out test split, plus per-sequence rank of known events (the operationally meaningful metric — does the comet surface at the top of its own day?).
- Runs are identified by config-and-data hashes (`run_<confighash>_data<datahash>_seed<n>`), so a score in the database is traceable to the exact code, config, data split, and seed that produced it.

---

## Scoring, Fusion & Prioritization

- Each checkpoint writes per-candidate scores under its own `run_id`.
- The primary ranking signal is **`fusion_mean_v1`** — the mean of sigmoid scores across the checkpoint pair (two independently seeded/data-split runs), damping single-checkpoint noise.
- Candidates are assigned a status by fusion score: **HIGH_PRIORITY**, **MEDIUM_PRIORITY**, or **LOW_PRIORITY** — this drives the Review queue.
- Scores are **uncalibrated ranking signals**, exposed as such everywhere in the UI and agent reports; they are not detection probabilities.

---

## Live Analysis

The Live page (and `/api/live/*`) pulls a recent day from NASA's rolling archive and runs the **entire pipeline** on it as a single background job:

- `GET /api/live/available` — the last 14 days with frame counts on the archive and local ingestion state (cached 10 minutes; probed with 6 parallel workers).
- `POST /api/live/run {"date": "YYYY-MM-DD"}` — starts the job (409 if one is already running).
- `GET /api/live/status` — state, current stage, stages done/total, rolling log tail, sequence ids, error.

Each of the 11 stages runs as a subprocess of the documented `python -m` CLI chain, so behavior is *identical* to running the stages by hand and every idempotency/config-hash guarantee holds. A failed run can simply be re-run: completed work is skipped via manifests and hashes, and the job resumes where it stopped.

---

## Archive-Resilience Engineering

The public archives are genuinely hostile input, and the pipeline is hardened against every failure mode observed in production:

| Archive behavior | Defense |
|---|---|
| First request per file stalls ~20+ s (server-side cache warming), retry succeeds fast | fail-fast timeout (`LASCO_HTTP_TIMEOUT_SECONDS`, 20 s in live mode) + retry with backoff; parallel workers overlap the stalls |
| Transient timeouts / connection resets | 3 download attempts per file; failures recorded in the manifest and retried on the next run |
| A few frames permanently unavailable | partial-failure tolerance — stages fail only when *nothing* succeeded; sequences tolerate gaps by design |
| Frames re-served under new file ids with duplicate DATE-OBS | timestamp dedup before tracking + zero-span velocity guard |
| File ids out of chronological order | all ordering and coverage math uses timestamps, never filename order |
| Occasional 512×512 frames in a 1024×1024 day | dimension check at registration; mismatched frames excluded with a warning |
| Malformed FITS headers (non-printable header cards) | astropy verification warnings tolerated; load validated on data + DATE-OBS presence |
| Truncated downloads | atomic `.part` → rename writes; SHA-256 recorded per file |

---

## Discovery Agent (LLM)

`backend/agent/discovery.py` — a two-node LangGraph behind `GET /api/candidates/{track_id}/report`:

1. **Fetch** the candidate's evidence package (track summary, Motion DNA, per-run + fusion scores, known-event attribution, caveats).
2. **Draft** a markdown report with Groq (`openai/gpt-oss-120b`) in a fixed format: *Assessment* (how comet-like and why), *Key evidence* (3–6 bullets citing concrete numbers from the package), *Caveats* (always including that scores are ranking signals and confirmation requires human review + astrometric verification).

Guardrails, by construction:

- The prompt forbids claiming discoveries and inventing numbers not present in the evidence.
- Known-event candidates are reported plainly as validation cases, not finds.
- **Failure-safe**: no `GROQ_API_KEY`, a network error, or any LLM failure degrades to a deterministic report assembled from the same evidence — the endpoint never breaks because the LLM is unavailable. Only a missing candidate raises (so the API can 404).

---

## Data Model

PostgreSQL via SQLAlchemy (`backend/db/models.py`), migrated with Alembic:

| Table | Contents |
|---|---|
| `missions` | instrument/mission registry (SOHO/LASCO) |
| `image_sequences` | one per contiguous day-sequence: instrument, start/end, frame count, status |
| `images` | one per FITS frame: paths, hashes, observation time, sequence membership |
| `candidates` | one per kept track: track id, timing, n_frames, Motion DNA features (JSON), label, pipeline status, human review verdict + notes + timestamp |
| `model_predictions` | one per (candidate, run): run id, model version, score — including the fusion pseudo-run |

Human review (`review`, `reviewer_notes`, `reviewed_at`) is a separate concern from pipeline `status` and is never touched by re-scoring.

---

## Failure-Safe Design

No single failure kills an analysis:

- **Download failures** → recorded per file, retried next run, tolerated if partial.
- **Bad frames** → QA-flagged (preprocess) or excluded (register), never fatal in small numbers.
- **Duplicate/out-of-order archive data** → handled structurally (see Archive Resilience).
- **LLM unavailable** → deterministic report fallback.
- **A stage crash mid-run** → all completed work persists (atomic writes + manifests + config hashes); re-running resumes from the failure point.
- **Live job errors** → surfaced via `/api/live/status` with the log tail; the API process never crashes.

---

## Provenance & Reproducibility

- Every pipeline stage embeds a **config hash** in its output; changing any parameter yields a new hash instead of silently overwriting.
- Raw files carry SHA-256 digests from download time.
- Track IDs, run IDs, and data-split hashes make every number in the database traceable to exact code + config + data.
- The full validation summary is reproducible offline with `python -m ml.validation_summary`.

---

## API Reference

FastAPI service (`backend/api/main.py`), interactive docs at `/docs`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | API + database health |
| `GET` | `/api/sequences` | all sequences with frame and candidate counts |
| `GET` | `/api/candidates` | ranked candidate list; filters: `status`, `label`, `sequence_id`, `q` (track-id search), `limit`, `offset`; sorted by fusion score |
| `GET` | `/api/candidates/{track_id}` | full candidate detail: features, per-run predictions, fusion score, review state |
| `GET` | `/api/candidates/{track_id}/evidence` | complete evidence package (track, DNA, scores, attribution, provenance) |
| `GET` | `/api/candidates/{track_id}/similar` | Motion-DNA nearest neighbors across the whole archive (z-scored L2) |
| `GET` | `/api/candidates/{track_id}/crops.png` | detection crop stack rendered as a PNG filmstrip (`scale` 1–8) — the frontend animates it as a flipbook |
| `GET` | `/api/candidates/{track_id}/report` | discovery-agent report (LLM or deterministic fallback) |
| `PATCH` | `/api/candidates/{track_id}/review` | set/clear human verdict and notes (validated verdict vocabulary, 10k-char note limit) |
| `GET` | `/api/statistics` | mission-wide statistics: candidates by status/label, sequences, model runs |
| `GET` | `/api/live/available` | recent archive days with frame counts + local ingestion state |
| `POST` | `/api/live/run` | start a full live pipeline job for a date |
| `GET` | `/api/live/status` | live job state, stage progress, log tail |

---

## The Interface

React + Vite mission-control UI (`frontend/`), six views on an animated starfield:

- **Overview** — mission statistics: candidate counts by priority and label, sequences processed, model runs.
- **Candidates (Explorer)** — ranked explorer with status/label filters, track-id search, and pagination; fusion scores surfaced on every row.
- **Candidate detail** — animated detection-crop flipbook, trajectory plot with Sun sight-line, full Motion DNA table, per-run + fusion scores, known-object attribution, provenance, the agent's discovery report, and the human review panel (verdict + notes).
- **Review queue** — high-priority tracks, best first — the human-in-the-loop workflow surface.
- **Archaeology** — Motion-DNA similarity search across the entire processed archive; querying a known comet returns the other known comets, the honest "trajectory archaeology" demo.
- **Live** — pick a recent archive day, launch the full pipeline, and watch stage-by-stage progress with the rolling log.

---

## Project Structure

```
SUNGRAZER AI/
├── backend/
│   ├── ingestion/
│   │   ├── ingest.py            # day/camera downloads (parallel, manifest, atomic)
│   │   ├── archive_client.py    # HTTP client, listing parser, retrying downloader
│   │   ├── manifest.py          # per-day download manifest
│   │   ├── persist.py           # files → images table
│   │   ├── sequences.py         # frames → day-sequences (gap-based grouping)
│   │   └── config.py            # env-driven archive/timeout/worker config
│   ├── pipeline/
│   │   ├── fits_io.py           # tolerant FITS loading, DATE-OBS parsing
│   │   ├── preprocess.py        # normalize, mask, per-frame QA
│   │   ├── register.py          # phase-correlation alignment (+ ECC)
│   │   ├── motion.py            # temporal-median subtraction, blob detection
│   │   ├── tracks.py            # association, coasting, dedup, persistence filter
│   │   └── motion_dna.py        # behavioral features + validation
│   ├── ml/
│   │   ├── crops.py             # 32×32 crop stacks per track
│   │   ├── net.py               # temporal_ranker_v1 (CNN + GRU, ~60k params)
│   │   ├── dataset.py           # splits, labels, loading
│   │   ├── train.py             # training runs (config/data-hashed run ids)
│   │   ├── metrics.py           # AP, per-sequence ranks
│   │   ├── score.py             # checkpoint scoring + fusion + priority status
│   │   └── validation_summary.py# reproduce the headline numbers
│   ├── dataset/
│   │   ├── candidates.py        # tracks → candidates table (label-preserving)
│   │   ├── evidence.py          # evidence package builder
│   │   ├── events.py            # Sungrazer Project known-event attribution
│   │   ├── labels.py            # positive/negative labeling
│   │   └── find_comet.py        # targeted recovery tooling
│   ├── agent/
│   │   └── discovery.py         # LangGraph + Groq report agent (failure-safe)
│   ├── api/
│   │   ├── main.py              # FastAPI: candidates, evidence, review, stats
│   │   └── live.py              # live pull-and-analyze background job
│   ├── db/
│   │   ├── models.py            # SQLAlchemy models
│   │   └── session.py           # engine/session (loads root .env)
│   ├── alembic/                 # schema migrations
│   └── tests/                   # 100+ offline tests (no network/GPU needed)
├── frontend/
│   └── src/
│       ├── pages/               # Overview, Explorer, Detail, Archaeology, Live
│       ├── components/          # Layout, ReviewPanel, Starfield, StatusBadge
│       └── lib/                 # API client
├── data/                        # raw FITS + processed outputs (gitignored)
└── docker-compose.yml           # PostgreSQL 17
```

---

## Technology Stack

### Backend
- Python 3.13, FastAPI + Uvicorn
- NumPy, OpenCV, Astropy (FITS), SciPy-free by design
- PyTorch (CUDA-accelerated scoring when available)
- PostgreSQL 17 (Docker) + SQLAlchemy 2 + Alembic
- Pydantic v2 request validation

### AI / Agent
- LangGraph + langchain-groq
- Groq LLM API — `openai/gpt-oss-120b` — narration only, deterministic fallback

### Frontend
- React 18 + Vite + TypeScript
- Animated canvas starfield, flipbook crop animation, SVG trajectory plots

---

## Running It

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

Configuration is environment-driven — copy `.env.example` to `.env` at the repo root (loaded automatically by every entry point). `GROQ_API_KEY` is optional: without it, agent reports use the deterministic fallback.

Useful knobs: `SOHO_DATA_ROOT`, `LASCO_HTTP_TIMEOUT_SECONDS`, `LASCO_DOWNLOAD_RETRIES`, `LASCO_DOWNLOAD_WORKERS`, `SEQUENCE_GAP_MINUTES`, `LASCO_HISTORICAL_BASE_URL`, `LASCO_RECENT_BASE_URL`.

### Processing data

Easiest: open the **Live** page and click a recent day.

Manually (per sequence), the documented CLI chain:

```
ingestion.ingest → ingestion.persist → ingestion.sequences
→ pipeline.preprocess → pipeline.register → pipeline.motion
→ pipeline.tracks → pipeline.motion_dna → ml.crops
→ dataset.candidates → ml.score
```

Every stage is an idempotent `python -m` CLI writing config-hashed outputs under `data/processed/` — safe to interrupt and re-run at any point.

---

## Testing

```bash
# from backend/
python -m pytest tests -q               # 100+ offline tests, no network/GPU needed

# Reproduce the validation summary
python -m ml.validation_summary
```

Coverage spans every pipeline stage (ingestion, FITS I/O, preprocessing, registration, motion, tracking, Motion DNA, ML, candidates/evidence, agent, API) with synthetic data — no network or GPU required.

---

## Data Sources

- **Historical (full mission):** `https://lasco-www.nrl.navy.mil/lz/level_05/` (NRL)
- **Recent (~2 weeks):** `https://umbra.nascom.nasa.gov/pub/lasco/lastimage/level_05/` (NASA SDAC)
- **Ground truth:** the [Sungrazer Project](https://sungrazer.nrl.navy.mil/) confirmed-comet lists

Both archives verified 2026-09-12; layout `<base>/<YYMMDD>/<c2|c3>/<8-digit-id>.fts`.

---

## Validation Results

From [`docs/validation.md`](docs/validation.md):

| Metric | Value |
|---|---|
| Day-sequences processed (2024, LASCO C3) | 41 |
| Moving-object tracks extracted | 34,994 |
| Confirmed Sungrazer events in the window | 15 |
| Events recovered by the pipeline | **10** |
| Rank of each recovered event within its own day-sequence | **#1 (all 10)** |
| Held-out test Average Precision | 0.4613 (7 positives vs 7,484 negatives) |

The five misses were not ranked low — their tracks did not survive extraction (too faint, too short, or lost at frame boundaries), which is a detection-recall limitation, not a ranking failure.

---

## Honest Limitations

- **Batch-oriented**: the Live page pulls on demand; there is no continuous polling daemon (by design for a local-first tool).
- **Coverage**: validated on LASCO C3, 2024, Kreutz-group events; other cameras/years/comet groups are untested.
- **Small positive set**: ~22 positive tracks total; every metric carries wide uncertainty, and seed-to-seed variation is real (hence checkpoint fusion).
- **Uncalibrated scores**: fusion scores rank; they are not probabilities.
- **Human confirmation required**: a candidate — however highly ranked — is a lead for human review and astrometric verification, never a claimed discovery.

---

## Future Improvements

- Scheduled/continuous live polling with alerting on high-priority candidates
- LASCO C2 support and multi-camera track linking
- More positives via additional validation years → better-calibrated scores
- Astrometric conversion (pixel tracks → sky coordinates) for direct Sungrazer Project report drafts
- Detection-recall work on the missed events (fainter thresholds, boundary handling)
- Model experiments already scaffolded: Motion-DNA feature fusion, transformer temporal encoders, regularization sweeps

---
