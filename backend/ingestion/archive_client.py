"""HTTP client for the public LASCO level-0.5 archives.

Both verified archives (NRL historical, NASA SDAC recent) share the layout:

    <base_url>/<YYMMDD>/<camera>/<NNNNNNNN>.fts

and serve plain HTML directory listings, so listing a day means parsing
anchor tags for ``*.fts`` names.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import date
from pathlib import Path

import requests

from ingestion import config

logger = logging.getLogger(__name__)

# Archive file names are 8-digit numeric identifiers, e.g. 32765039.fts
_FTS_LINK_RE = re.compile(r'href="(\d{8}\.fts)"', re.IGNORECASE)


def day_url(base_url: str, obs_date: date, camera: str) -> str:
    """Build the directory URL for one observation day and camera."""
    if camera not in config.CAMERAS:
        raise ValueError(f"Unknown LASCO camera {camera!r}; expected one of {config.CAMERAS}")
    return f"{base_url}/{obs_date:%y%m%d}/{camera}/"


def parse_listing(html: str) -> list[str]:
    """Extract sorted, de-duplicated .fts file names from a directory listing."""
    return sorted(set(_FTS_LINK_RE.findall(html)))


def list_day_files(obs_date: date, camera: str, base_url: str) -> list[str]:
    """List .fts files for one day/camera. Returns [] when the day directory
    does not exist (404), which callers use to fall back to another archive."""
    url = day_url(base_url, obs_date, camera)
    response = requests.get(url, timeout=config.HTTP_TIMEOUT_SECONDS)
    if response.status_code == 404:
        logger.info("No archive directory at %s (404)", url)
        return []
    response.raise_for_status()
    files = parse_listing(response.text)
    logger.info("Listed %d .fts files at %s", len(files), url)
    return files


def download_file(url: str, dest: Path) -> str:
    """Download one file atomically and return its SHA-256 hex digest.

    Writes to ``<dest>.part`` first and renames on success so an interrupted
    download never leaves a truncated file that looks complete. Retries
    transient failures with linear backoff.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, config.DOWNLOAD_RETRIES + 1):
        try:
            sha256 = hashlib.sha256()
            with requests.get(url, stream=True, timeout=config.HTTP_TIMEOUT_SECONDS) as response:
                response.raise_for_status()
                with open(part, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)
                        sha256.update(chunk)
            part.replace(dest)
            return sha256.hexdigest()
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            logger.warning("Download attempt %d/%d failed for %s: %s",
                           attempt, config.DOWNLOAD_RETRIES, url, exc)
            part.unlink(missing_ok=True)
            if attempt < config.DOWNLOAD_RETRIES:
                time.sleep(attempt)  # ponytail: linear backoff; make exponential if archives throttle
    raise RuntimeError(f"Failed to download {url} after {config.DOWNLOAD_RETRIES} attempts") from last_error
