"""Engine and session factory, configured through DATABASE_URL.

Default matches the docker-compose PostgreSQL service. Nothing here opens a
connection at import time; the engine connects lazily on first use.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DEFAULT_DATABASE_URL = "postgresql+psycopg://sungrazer:sungrazer@localhost:5432/sungrazer"


def database_url() -> str:
    """Return the configured database URL (env var DATABASE_URL wins)."""
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


engine = create_engine(database_url(), pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
