"""Atomic file writes that survive Windows file-lock transients.

On Windows, ``Path.replace()`` can transiently fail with PermissionError
(WinError 5/32) when an indexer or antivirus briefly holds the target.
Observed in practice on 2026-09-13 during a 112-frame registration run.
All pipeline manifest/output writes go through these helpers, which retry
the rename briefly before giving up.
"""

from __future__ import annotations

import time
from pathlib import Path

_RETRIES = 5
_RETRY_DELAY_S = 0.2


def atomic_replace(tmp: Path, dest: Path) -> None:
    """``tmp.replace(dest)`` with retries for transient Windows locks."""
    for attempt in range(_RETRIES):
        try:
            tmp.replace(dest)
            return
        except PermissionError:
            if attempt == _RETRIES - 1:
                raise
            time.sleep(_RETRY_DELAY_S * (attempt + 1))


def atomic_write_text(dest: Path, text: str) -> None:
    """Write text to ``dest`` atomically (tmp file + retried rename)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    atomic_replace(tmp, dest)


def atomic_write_bytes_via(dest: Path, writer) -> None:
    """Call ``writer(file_handle)`` on a tmp file, then rename atomically."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    with open(tmp, "wb") as fh:
        writer(fh)
    atomic_replace(tmp, dest)
