"""Scientific processing pipeline stages (FITS I/O, preprocessing, ...).

Sits between raw ingested files (backend/ingestion) and the database layer
(backend/db). Each stage is importable and testable on its own.
"""
