"""Per-day ingestion manifest for idempotency and resume.

One ``manifest.json`` per ``<raw>/<YYMMDD>/<camera>/`` directory records every
file the collector has fully downloaded and validated. Re-running ingestion
for the same day skips completed entries, satisfying the idempotency and
resumability requirements in docs/claude.md sections 23 and 25.

PostgreSQL is the eventual system of record; this local manifest is the
prototype-stage persistence explicitly allowed by claude.md section 23
("PostgreSQL and/or well-defined local data manifests").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "manifest.json"

STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"


@dataclass
class Manifest:
    directory: Path
    entries: dict[str, dict] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return self.directory / MANIFEST_NAME

    @classmethod
    def load(cls, directory: Path) -> "Manifest":
        manifest = cls(directory=directory)
        if manifest.path.exists():
            with open(manifest.path, encoding="utf-8") as fh:
                manifest.entries = json.load(fh)
        return manifest

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.entries, fh, indent=2, sort_keys=True)
        tmp.replace(self.path)

    def is_complete(self, filename: str) -> bool:
        return self.entries.get(filename, {}).get("status") == STATUS_COMPLETE

    def record(self, filename: str, *, status: str, source_url: str,
               sha256: str | None = None, size_bytes: int | None = None,
               date_obs: str | None = None, error: str | None = None) -> None:
        entry = {
            "status": status,
            "source_url": source_url,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        if sha256 is not None:
            entry["sha256"] = sha256
        if size_bytes is not None:
            entry["size_bytes"] = size_bytes
        if date_obs is not None:
            entry["date_obs"] = date_obs
        if error is not None:
            entry["error"] = error
        self.entries[filename] = entry
