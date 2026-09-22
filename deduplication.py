"""
Layer 0: Deduplication & Fuzzy Description Hashing for the Unified AI Job Pipeline.

Provides O(1) deduplication via:
  - Canonical URL matching (tracking query params stripped)
  - Normalized Title + Company hashing
  - Fuzzy / Anchored description hashing to catch recycled enterprise job postings (Workday ATS, etc.)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from turso_python import AsyncTursoConnection

from filters import extract_anchored_description, generate_desc_hash
from database import (
    get_turso_client,
    is_job_seen as _db_is_job_seen,
    is_job_duplicate as _db_is_job_duplicate,
    is_new_job_async as _db_is_new_job_async,
    is_new_job as _db_is_new_job,
    make_job_id as _db_make_job_id,
    title_company_hash as _db_title_company_hash,
)

log = logging.getLogger("deduplication")

# Direct alias exports
title_company_hash = _db_title_company_hash
make_job_id = _db_make_job_id
is_job_seen = _db_is_job_seen
is_job_duplicate = _db_is_job_duplicate
is_new_job_async = _db_is_new_job_async
is_new_job = _db_is_new_job
is_new_job_sync = _db_is_new_job

__all__ = [
    "extract_anchored_description",
    "generate_desc_hash",
    "title_company_hash",
    "make_job_id",
    "is_job_seen",
    "is_job_duplicate",
    "is_new_job_async",
    "is_new_job",
    "is_new_job_sync",
]
