"""Parse Sungrazer-project confirmation lists into structured comet events.

Usage:
    python -m dataset.events --file PATH [--camera c3] [--group Kreutz]
                             [--min-days 2]

Source: https://sungrazer.nrl.navy.mil/index.php/confirmation_lists —
per-year text files of confirmed SOHO comet discoveries (verified live
2026-09-13; local copies under ``data/dataset/sources/``). Line format
(whitespace-separated, hand-maintained by NRL so tolerance is required):

    5011  Apr30,24 20:53:41  W.Boonplod  C3,C2  Kreutz  Apr30-May01,24

Fields: SOHO number (optional ``*`` flag), report date + time (time
formats vary: ``20:53:41``, ``10:52PM`` — not used by us), discoverer(s),
cameras, group, observation date range (may cross months:
``Jun29-Jul01,24``), optional trailing notes (e.g. ``cor2``).

The parser anchors on the cameras token (the only token matching the
camera pattern) rather than fixed columns, and is deliberately lenient:
unparseable lines are collected, logged, and never silently dropped.

NRL states the lists contain mistakes and are not exhaustive; they are a
candidate index for building the labeled dataset, not ground truth —
every selected event is verified against the actual imagery before a
track is labeled positive.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}

_CAMERAS_RE = re.compile(r"^C[23][A-Z]?(,C[23][A-Z]?)*$")
# Jan25,24 | Jan25-27,24 | Apr30-May01,24
_OBS_RE = re.compile(
    r"^([A-Z][a-z]{2})(\d{2})(?:-([A-Z][a-z]{2})?(\d{2}))?,(\d{2})$")


@dataclass(frozen=True)
class CometEvent:
    """One confirmed comet from a Sungrazer confirmation list."""

    soho_number: int
    flagged: bool               # '*' on the number (annotated entry)
    discoverers: str
    cameras: tuple[str, ...]    # e.g. ("C3", "C2")
    group: str                  # Kreutz | Marsden | Meyer | Kracht | NonGrp | ...
    obs_start: date
    obs_end: date
    source_line: str

    @property
    def obs_days(self) -> list[date]:
        n = (self.obs_end - self.obs_start).days + 1
        return [self.obs_start + timedelta(days=i) for i in range(n)]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["obs_start"] = self.obs_start.isoformat()
        d["obs_end"] = self.obs_end.isoformat()
        d["cameras"] = list(self.cameras)
        return d


def _parse_obs_dates(token: str) -> tuple[date, date] | None:
    m = _OBS_RE.match(token)
    if not m:
        return None
    mon1, day1, mon2, day2, yy = m.groups()
    year = 2000 + int(yy)
    start = date(year, _MONTHS[mon1], int(day1))
    if day2 is None:
        return start, start
    end_month = _MONTHS[mon2] if mon2 else start.month
    end_year = year + 1 if end_month < start.month else year  # Dec->Jan
    return start, date(end_year, end_month, int(day2))


def parse_line(line: str) -> CometEvent | None:
    """Parse one confirmation-list line; None if it isn't an event line."""
    tokens = line.split()
    if len(tokens) < 6 or not re.match(r"^\d+\*?$", tokens[0]):
        return None
    camera_idx = next((i for i, t in enumerate(tokens)
                       if _CAMERAS_RE.match(t)), None)
    if camera_idx is None or camera_idx + 2 >= len(tokens) + 1:
        return None
    obs = _parse_obs_dates(tokens[camera_idx + 2]) \
        if camera_idx + 2 < len(tokens) else None
    if obs is None:
        return None
    return CometEvent(
        soho_number=int(tokens[0].rstrip("*")),
        flagged=tokens[0].endswith("*"),
        discoverers=" ".join(tokens[3:camera_idx]),
        cameras=tuple(tokens[camera_idx].split(",")),
        group=tokens[camera_idx + 1],
        obs_start=obs[0],
        obs_end=obs[1],
        source_line=line.rstrip(),
    )


def load_events(path: Path) -> tuple[list[CometEvent], list[str]]:
    """Parse a confirmation-list file. Returns (events, unparseable_lines)."""
    events, failures = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = parse_line(line)
        if event is None:
            failures.append(line.rstrip())
        else:
            events.append(event)
    if failures:
        logger.warning("%d unparseable lines in %s", len(failures), path)
    return events, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List comet events from a Sungrazer confirmation file")
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--camera", help="require this camera (e.g. c3)")
    parser.add_argument("--group", help="require this group (e.g. Kreutz)")
    parser.add_argument("--min-days", type=int, default=1,
                        help="minimum observation-span days")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args(argv)

    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
    events, failures = load_events(args.file)
    selected = [
        e for e in events
        if (not args.camera or args.camera.upper() in e.cameras)
        and (not args.group or e.group.lower() == args.group.lower())
        and len(e.obs_days) >= args.min_days
    ]
    if args.json:
        print(json.dumps([e.to_dict() for e in selected], indent=2))
    else:
        for e in selected:
            print(f"SOHO-{e.soho_number}  {e.obs_start} .. {e.obs_end} "
                  f"({len(e.obs_days)}d)  {','.join(e.cameras):9s} {e.group}")
        print(f"\n{len(selected)} selected / {len(events)} parsed / "
              f"{len(failures)} unparseable")
    for line in failures:
        logger.warning("UNPARSED: %s", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
