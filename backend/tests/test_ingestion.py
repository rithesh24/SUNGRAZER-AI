"""Offline unit tests for the ingestion prototype (no network required).

Run:  python tests/test_ingestion.py   (or pytest tests/)
"""

import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion import archive_client
from ingestion.manifest import Manifest, STATUS_COMPLETE, STATUS_FAILED


SAMPLE_LISTING = """
<html><body><table>
<tr><td><a href="32765039.fts">32765039.fts</a></td><td>2.0M</td></tr>
<tr><td><a href="32765039.jpg">32765039.jpg</a></td><td>1.8K</td></tr>
<tr><td><a href="32765040.fts">32765040.fts</a></td><td>2.0M</td></tr>
<tr><td><a href="32765040.fts">32765040.fts</a></td><td>duplicate link</td></tr>
<tr><td><a href="manifest.json">manifest.json</a></td></tr>
<tr><td><a href="../">Parent Directory</a></td></tr>
</table></body></html>
"""


def test_parse_listing():
    files = archive_client.parse_listing(SAMPLE_LISTING)
    assert files == ["32765039.fts", "32765040.fts"], files


def test_parse_listing_empty():
    assert archive_client.parse_listing("<html>404 not found</html>") == []


def test_day_url():
    url = archive_client.day_url("https://example.org/lz/level_05", date(2024, 1, 1), "c3")
    assert url == "https://example.org/lz/level_05/240101/c3/", url
    try:
        archive_client.day_url("https://example.org", date(2024, 1, 1), "c9")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid camera should raise ValueError")


def test_manifest_roundtrip_and_skip():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        m = Manifest.load(directory)
        assert m.entries == {}
        m.record("a.fts", status=STATUS_COMPLETE, source_url="http://x/a.fts",
                 sha256="ab", size_bytes=10, date_obs="2024-01-01T00:00:00")
        m.record("b.fts", status=STATUS_FAILED, source_url="http://x/b.fts", error="boom")
        m.save()

        reloaded = Manifest.load(directory)
        assert reloaded.is_complete("a.fts")
        assert not reloaded.is_complete("b.fts")  # failed entries are retried
        assert not reloaded.is_complete("c.fts")
        assert reloaded.entries["b.fts"]["error"] == "boom"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("All ingestion unit tests passed.")
