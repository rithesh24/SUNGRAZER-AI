"""Environment-driven configuration for LASCO ingestion.

All values can be overridden through environment variables so nothing
environment-specific is hard-coded (see docs/claude.md section 14).
"""

import os
from pathlib import Path

# Root directory for all locally stored data (raw + processed).
DATA_ROOT = Path(os.environ.get("SOHO_DATA_ROOT", "data"))

# Raw LASCO level-0.5 files live under: <RAW_DIR>/<YYMMDD>/<camera>/
RAW_DIR = DATA_ROOT / "raw" / "soho" / "lasco"

# Full-mission historical archive (NRL). Verified 2026-09-12:
# https://lasco-www.nrl.navy.mil/lz/level_05/240101/c3/ lists NNNNNNNN.fts files.
HISTORICAL_BASE_URL = os.environ.get(
    "LASCO_HISTORICAL_BASE_URL",
    "https://lasco-www.nrl.navy.mil/lz/level_05",
)

# Rolling ~2-week archive (NASA SDAC), same layout. Verified 2026-09-12.
RECENT_BASE_URL = os.environ.get(
    "LASCO_RECENT_BASE_URL",
    "https://umbra.nascom.nasa.gov/pub/lasco/lastimage/level_05",
)

HTTP_TIMEOUT_SECONDS = float(os.environ.get("LASCO_HTTP_TIMEOUT_SECONDS", "60"))
DOWNLOAD_RETRIES = int(os.environ.get("LASCO_DOWNLOAD_RETRIES", "3"))

# Maximum time between consecutive frames of one image_sequence. Observed
# nominal cadence: ~12 min (C3), ~20 min (C2); 60 min tolerates a few missed
# frames without merging across real observation breaks.
SEQUENCE_GAP_MINUTES = float(os.environ.get("SEQUENCE_GAP_MINUTES", "60"))

# LASCO cameras with data in the level-0.5 archive. C2/C3 are the
# coronagraphs relevant to sungrazer detection.
CAMERAS = ("c2", "c3")
