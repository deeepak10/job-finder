"""
Turso Cloud Database layer for the Unified Autonomous AI Job Pipeline.

Handles:
  * Asynchronous connection management via AsyncTursoConnection (turso-python)
  * Schema creation (idempotent)
  * Deterministic job_id and title+company hashing
  * Layer 0 (O(1) Deduplication) querying Turso for URL or hash(title + company)
  * Async persistence of evaluated job postings and alert statuses
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

from turso_python import AsyncTursoConnection
from turso_python.response_parser import TursoResponseParser

import config

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Schema Specification (v2)
# --------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS job_postings (
    job_id              TEXT PRIMARY KEY,
    title               TEXT    NOT NULL,
    company             TEXT    NOT NULL,
    location            TEXT,
    platform            TEXT    NOT NULL,
    url                 TEXT    NOT NULL,
    description         TEXT,
    ai_score            INTEGER,
    visa_sponsorship    TEXT,
    date_found          TEXT,
    alert_sent          INTEGER DEFAULT 0,
    status              TEXT DEFAULT 'active',
    tier                TEXT DEFAULT 'strict',
    match_reason        TEXT DEFAULT ''
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_jobs_url ON job_postings (url);",
    "CREATE INDEX IF NOT EXISTS idx_jobs_date_found ON job_postings (date_found DESC);",
    "CREATE INDEX IF NOT EXISTS idx_jobs_score ON job_postings (ai_score DESC);",
    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON job_postings (status);",
    "CREATE INDEX IF NOT EXISTS idx_jobs_tier ON job_postings (tier);",
]


# --------------------------------------------------------------------------
# Connection Management
# --------------------------------------------------------------------------

def get_turso_client(
    database_url: Optional[str] = None,
    auth_token: Optional[str] = None,
    timeout: int = 30,
) -> AsyncTursoConnection:
    """Create an AsyncTursoConnection instance using configuration or environment.

    Args:
        database_url: Optional explicit LibSQL database URL. If omitted, falls back
            to config.TURSO_DATABASE_URL or the TURSO_DATABASE_URL environment variable.
        auth_token: Optional explicit authentication token. If omitted, falls back
            to config.TURSO_AUTH_TOKEN or the TURSO_AUTH_TOKEN environment variable.
        timeout: HTTP request timeout in seconds (default: 30).

    Returns:
        An initialized AsyncTursoConnection instance configured with automatic retries.

    Raises:
        ValueError: If TURSO_DATABASE_URL is not provided or configured in the environment.
    """
    url = database_url or config.TURSO_DATABASE_URL or os.getenv("TURSO_DATABASE_URL")
    token = auth_token or config.TURSO_AUTH_TOKEN or os.getenv("TURSO_AUTH_TOKEN")

    if not url:
        raise ValueError("TURSO_DATABASE_URL must be provided or configured in .env")

    return AsyncTursoConnection(
        database_url=url,
        auth_token=token,
        timeout=timeout,
        retries=2,
    )


def parse_turso_rows(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a raw Turso JSON query response into a list of typed dictionary rows.

    Args:
        response: Raw JSON response dictionary returned by AsyncTursoConnection.execute_query.

    Returns:
        List of dictionaries where each item maps column names to deserialized values.
    """
    normalized = TursoResponseParser.normalize_response(response)
    columns = normalized.get("columns", [])
    raw_rows = normalized.get("rows", [])
    result: list[dict[str, Any]] = []
    for row in raw_rows:
        row_dict: dict[str, Any] = {}
        for col, cell in zip(columns, row):
            if isinstance(cell, dict):
                if cell.get("type") == "null":
                    row_dict[col] = None
                elif "value" in cell:
                    row_dict[col] = cell["value"]
                else:
                    row_dict[col] = None
            else:
                row_dict[col] = cell
        result.append(row_dict)
    return result


# --------------------------------------------------------------------------
# Schema Initialization
# --------------------------------------------------------------------------

async def init_db(client: Optional[AsyncTursoConnection] = None) -> None:
    """Initialize the Turso database schema and secondary indexes idempotently.

    Creates the `job_postings` table and performance indexes (`idx_jobs_url`,
    `idx_jobs_date_found`, `idx_jobs_score`) if they do not exist. Cleans up
    legacy columns from prior schema iterations.

    Args:
        client: Optional shared AsyncTursoConnection. If omitted, a scoped client is
            created and automatically closed upon completion.
    """
    close_client = False
    if client is None:
        client = get_turso_client()
        close_client = True

    try:
        log.info("Initializing Turso database schema...")
        await client.execute_query(SCHEMA)
        for idx_sql in INDEXES:
            await client.execute_query(idx_sql)

        # Ensure description, status, tier, and match_reason columns exist for schema migrations
        for col_def in (
            ("description", "TEXT"),
            ("status", "TEXT DEFAULT 'active'"),
            ("tier", "TEXT DEFAULT 'strict'"),
            ("match_reason", "TEXT DEFAULT ''"),
        ):
            try:
                await client.execute_query(f"ALTER TABLE job_postings ADD COLUMN {col_def[0]} {col_def[1]};")
                log.info("Added column '%s' to job_postings.", col_def[0])
            except Exception:
                pass  # Column already exists

        # Drop legacy columns if present in existing table
        for legacy_col in ("ai_reasoning", "portfolio_highlight", "outreach_message"):
            try:
                await client.execute_query(f"ALTER TABLE job_postings DROP COLUMN {legacy_col};")
                log.info("Dropped legacy column '%s' from job_postings.", legacy_col)
            except Exception:
                pass

        log.info("Turso database initialized successfully.")
    finally:
        if close_client and client.session:
            await client.session.close()


# --------------------------------------------------------------------------
# Hashing & Job Identity
# --------------------------------------------------------------------------

_WHITESPACE = re.compile(r"\s+")
_TRACKING_PARAMS = ("?", "#")


def _normalize(value: str) -> str:
    return _WHITESPACE.sub(" ", (value or "").strip().lower())


def _strip_url(url: str) -> str:
    """Drop query parameters and trailing fragments to canonicalize."""
    clean = (url or "").strip()
    for marker in _TRACKING_PARAMS:
        clean = clean.split(marker, 1)[0]
    return clean.rstrip("/").lower()


def title_company_hash(title: str, company: str) -> str:
    """Deterministic hash of (title + company) used for Layer 0 dedup."""
    raw = f"{_normalize(title)}|{_normalize(company)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def make_job_id(platform: str, url: str = "", title: str = "", company: str = "") -> str:
    """Deterministic 16-character job ID.

    Prefers hash(title + company) combined with platform/url to avoid duplicates.
    """
    clean_u = _strip_url(url)
    tc_hash = title_company_hash(title, company)
    if clean_u:
        raw = f"{_normalize(platform)}|{clean_u}|{tc_hash}"
    else:
        raw = f"{_normalize(platform)}|{tc_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Layer 0 (O(1) Deduplication)
# --------------------------------------------------------------------------

async def is_job_seen(
    url_or_id: str = "",
    title: str = "",
    company: str = "",
    client: Optional[AsyncTursoConnection] = None,
) -> bool:
    """Execute Layer 0 O(1) deduplication check against Turso Cloud.

    Queries the primary key (`job_id`) and indexed `url` column to determine
    if a candidate job has already been evaluated or stored in prior runs.

    Args:
        url_or_id: Job posting URL or pre-computed unique identifier.
        title: Job posting title (used with company to derive hash).
        company: Hiring organization name.
        client: Optional shared AsyncTursoConnection instance.

    Returns:
        True if the job exists in the database; False if fresh or on query error (fail-open).
    """
    clean_url = _strip_url(url_or_id) if url_or_id.startswith("http") else ""
    tc_hash = title_company_hash(title, company) if (title and company) else ""
    target_id = url_or_id if not clean_url else tc_hash

    # Look up by primary key or indexed URL
    query = "SELECT 1 FROM job_postings WHERE url = ? OR job_id = ? OR job_id = ? LIMIT 1"
    args = [clean_url or url_or_id, target_id or url_or_id, tc_hash or url_or_id]

    close_client = False
    if client is None:
        client = get_turso_client()
        close_client = True

    try:
        res = await client.execute_query(query, args)
        rows = TursoResponseParser.extract_rows(res)
        return len(rows) > 0
    except Exception as exc:
        log.warning("Turso deduplication check error (%s); failing open: %s", url_or_id, exc)
        return False
    finally:
        if close_client and client.session:
            await client.session.close()


async def is_new_job_async(
    url_or_id: str = "",
    title: str = "",
    company: str = "",
    client: Optional[AsyncTursoConnection] = None,
) -> bool:
    """Async inverse of is_job_seen: True if the job is fresh and should be processed."""
    seen = await is_job_seen(url_or_id=url_or_id, title=title, company=company, client=client)
    return not seen


def is_new_job(url_or_id: str = "", title: str = "", company: str = "") -> bool:
    """Synchronous interface matching scrapers/base.py signature."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, is_new_job_async(url_or_id, title, company)).result()
    else:
        return asyncio.run(is_new_job_async(url_or_id, title, company))


is_new_job_sync = is_new_job



# --------------------------------------------------------------------------
# Insert & Update Operations
# --------------------------------------------------------------------------

async def add_job(
    job: dict[str, Any],
    client: Optional[AsyncTursoConnection] = None,
) -> bool:
    """Persist an evaluated job posting record into Turso Cloud SQLite.

    Inserts or updates the job posting with its metadata, AI match score,
    visa sponsorship status, and alert state.

    Args:
        job: Dictionary containing job fields: title, company, platform, url,
            and optional location, ai_score, visa_sponsorship, date_found.
        client: Optional shared AsyncTursoConnection instance.

    Returns:
        True if the record was inserted/updated successfully, False otherwise.

    Raises:
        ValueError: If any mandatory field (title, company, platform, url) is missing.
    """
    required = ("title", "company", "platform", "url")
    missing = [k for k in required if not job.get(k)]
    if missing:
        raise ValueError(f"Job is missing required field(s): {', '.join(missing)}")

    title = job["title"].strip()
    company = job["company"].strip()
    url = job["url"].strip()
    platform = job["platform"].strip().lower()
    location = (job.get("location") or "").strip()

    # Determine job_id: prefer hash(title + company) as primary identity
    job_id = job.get("job_id") or title_company_hash(title, company)
    description = (job.get("description") or "").strip()
    ai_score = job.get("ai_score")
    visa_sponsorship = job.get("visa_sponsorship") or ""
    date_found = job.get("date_found") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    alert_sent = int(bool(job.get("alert_sent", 0)))
    status = (job.get("status") or "active").strip()
    tier = (job.get("tier") or "strict").strip().lower()
    match_reason = (job.get("match_reason") or "").strip()

    sql = """
    INSERT OR REPLACE INTO job_postings (
        job_id, title, company, location, platform, url,
        description, ai_score, visa_sponsorship, date_found,
        alert_sent, status, tier, match_reason
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    args = [
        job_id,
        title,
        company,
        location,
        platform,
        url,
        description,
        ai_score,
        visa_sponsorship,
        date_found,
        alert_sent,
        status,
        tier,
        match_reason,
    ]

    close_client = False
    if client is None:
        client = get_turso_client()
        close_client = True

    try:
        res = await client.execute_query(sql, args)
        # Check affected row count
        if res.get("results") and res["results"][0].get("response"):
            result_meta = res["results"][0]["response"].get("result", {})
            return result_meta.get("affected_row_count", 0) > 0
        return True
    except Exception as exc:
        log.error("Failed to insert job '%s' at %s into Turso: %s", title, company, exc)
        return False
    finally:
        if close_client and client.session:
            await client.session.close()


async def get_deferred_jobs(
    limit: int = 50,
    client: Optional[AsyncTursoConnection] = None,
) -> list[dict[str, Any]]:
    """Retrieve jobs with status 'deferred' to be re-evaluated during morning runs.

    Args:
        limit: Maximum number of deferred jobs to retrieve (default: 50).
        client: Optional shared AsyncTursoConnection instance.

    Returns:
        List of dictionaries containing deferred job records.
    """
    sql = "SELECT * FROM job_postings WHERE status = 'deferred' ORDER BY date_found ASC LIMIT ?"
    close_client = False
    if client is None:
        client = get_turso_client()
        close_client = True

    try:
        res = await client.execute_query(sql, [limit])
        return parse_turso_rows(res)
    except Exception as exc:
        log.warning("Failed to query deferred jobs from Turso: %s", exc)
        return []
    finally:
        if close_client and client.session:
            await client.session.close()


async def get_pending_groq_jobs(
    limit: int = 50,
    client: Optional[AsyncTursoConnection] = None,
) -> list[dict[str, Any]]:
    """Retrieve jobs pending Groq verification from Turso Cloud SQLite.

    Used when Groq was offline during the broad candidate gatekeeper phase,
    and OpenRouter temporarily accepted the role. These jobs require secondary
    Groq verification before alerting to eliminate false positives.

    Args:
        limit: Maximum number of pending jobs to fetch.
        client: Optional shared AsyncTursoConnection instance.

    Returns:
        List of dictionaries containing pending Groq verification job records.
    """
    sql = "SELECT * FROM job_postings WHERE status = 'pending_groq_verification' ORDER BY date_found ASC LIMIT ?"
    close_client = False
    if client is None:
        client = get_turso_client()
        close_client = True

    try:
        res = await client.execute_query(sql, [limit])
        return parse_turso_rows(res)
    except Exception as exc:
        log.warning("Failed to query pending_groq_verification jobs from Turso: %s", exc)
        return []
    finally:
        if close_client and client.session:
            await client.session.close()


async def update_job_status(
    job_id: str,
    status: str,
    ai_score: Optional[int] = None,
    visa_sponsorship: Optional[str] = None,
    alert_sent: Optional[int] = None,
    tier: Optional[str] = None,
    match_reason: Optional[str] = None,
    client: Optional[AsyncTursoConnection] = None,
) -> bool:
    """Update status, score, visa sponsorship, and alert status for an existing job posting.

    Args:
        job_id: Unique primary key of the job posting.
        status: New status (e.g., 'active', 'consensus_passed', 'deferred', 'rejected').
        ai_score: Optional integer score.
        visa_sponsorship: Optional visa sponsorship string.
        alert_sent: Optional flag (0 or 1).
        tier: Optional tier string ('strict' or 'broad').
        match_reason: Optional 1-sentence match explanation.
        client: Optional shared AsyncTursoConnection instance.

    Returns:
        True if updated successfully, False otherwise.
    """
    updates = ["status = ?"]
    args: list[Any] = [status]

    if ai_score is not None:
        updates.append("ai_score = ?")
        args.append(ai_score)
    if visa_sponsorship is not None:
        updates.append("visa_sponsorship = ?")
        args.append(visa_sponsorship)
    if alert_sent is not None:
        updates.append("alert_sent = ?")
        args.append(int(bool(alert_sent)))
    if tier is not None:
        updates.append("tier = ?")
        args.append(tier.strip().lower())
    if match_reason is not None:
        updates.append("match_reason = ?")
        args.append(match_reason.strip())

    args.append(job_id)
    sql = f"UPDATE job_postings SET {', '.join(updates)} WHERE job_id = ?"

    close_client = False
    if client is None:
        client = get_turso_client()
        close_client = True

    try:
        res = await client.execute_query(sql, args)
        return bool(res.get("rows_affected", 0) > 0 or not res.get("error"))
    except Exception as exc:
        log.warning("Failed to update status for job '%s': %s", job_id, exc)
        return False
    finally:
        if close_client and client.session:
            await client.session.close()


async def mark_alert_sent(
    job_id: str,
    client: Optional[AsyncTursoConnection] = None,
) -> None:
    """Mark a job record as alerted after Discord notification succeeds.

    Args:
        job_id: The unique primary key of the job posting.
        client: Optional shared AsyncTursoConnection instance.
    """
    close_client = False
    if client is None:
        client = get_turso_client()
        close_client = True

    try:
        await client.execute_query(
            "UPDATE job_postings SET alert_sent = 1 WHERE job_id = ?",
            [job_id],
        )
    except Exception as exc:
        log.warning("Failed to update alert_sent for %s: %s", job_id, exc)
    finally:
        if close_client and client.session:
            await client.session.close()


async def get_job(
    job_id: str,
    client: Optional[AsyncTursoConnection] = None,
) -> Optional[dict[str, Any]]:
    """Retrieve a single job posting record from Turso Cloud by its job_id.

    Args:
        job_id: The unique 16-character identifier of the job posting.
        client: Optional shared AsyncTursoConnection instance.

    Returns:
        Job record dictionary if found, or None if no record matches.
    """
    close_client = False
    if client is None:
        client = get_turso_client()
        close_client = True

    try:
        res = await client.execute_query(
            "SELECT * FROM job_postings WHERE job_id = ? LIMIT 1",
            [job_id],
        )
        parsed = parse_turso_rows(res)
        return parsed[0] if parsed else None
    finally:
        if close_client and client.session:
            await client.session.close()


async def get_stats(client: Optional[AsyncTursoConnection] = None) -> dict[str, Any]:
    """Retrieve comprehensive aggregation metrics from Turso Cloud Database.

    Args:
        client: Optional shared AsyncTursoConnection instance.

    Returns:
        Dictionary containing total jobs, high matches, alerts dispatched, and platform breakdown.
    """
    close_client = False
    if client is None:
        client = get_turso_client()
        close_client = True

    try:
        total_res = await client.execute_query("SELECT COUNT(*) AS total FROM job_postings")
        total_rows = parse_turso_rows(total_res)
        total = int(total_rows[0]["total"]) if total_rows else 0

        platform_res = await client.execute_query(
            "SELECT platform, COUNT(*) AS n FROM job_postings GROUP BY platform"
        )
        by_platform = {
            r["platform"]: int(r["n"]) for r in parse_turso_rows(platform_res) if r.get("platform")
        }

        high_score_res = await client.execute_query(
            "SELECT COUNT(*) AS high_scores FROM job_postings WHERE ai_score >= 70"
        )
        high_score_rows = parse_turso_rows(high_score_res)
        high_scores = int(high_score_rows[0]["high_scores"]) if high_score_rows else 0

        alerted_res = await client.execute_query(
            "SELECT COUNT(*) AS alerted FROM job_postings WHERE alert_sent = 1"
        )
        alerted_rows = parse_turso_rows(alerted_res)
        alerted = int(alerted_rows[0]["alerted"]) if alerted_rows else 0

        deferred_res = await client.execute_query(
            "SELECT COUNT(*) AS deferred FROM job_postings WHERE status = 'deferred'"
        )
        deferred_rows = parse_turso_rows(deferred_res)
        deferred = int(deferred_rows[0]["deferred"]) if deferred_rows else 0

        return {
            "total_jobs": total,
            "high_match_jobs (>=70)": high_scores,
            "alerts_dispatched": alerted,
            "deferred_jobs": deferred,
            "by_platform": by_platform,
        }
    finally:
        if close_client and client.session:
            await client.session.close()


# --------------------------------------------------------------------------
# Maintenance & Garbage Collection
# --------------------------------------------------------------------------

async def run_garbage_collection_async(
    client: Optional[AsyncTursoConnection] = None,
) -> int:
    """Nullifies heavy text fields for rejected jobs older than 30 days to save Turso storage."""
    close_client = False
    if client is None:
        try:
            client = get_turso_client()
            close_client = True
        except Exception as e:
            log.warning("Could not connect to Turso for garbage collection: %s", e)
            return 0

    sql = """
    UPDATE job_postings 
    SET description = NULL, match_reason = NULL 
    WHERE status = 'rejected' 
      AND date_found < datetime('now', '-30 days');
    """
    try:
        res = await client.execute_query(sql)
        rows_affected = 0
        if isinstance(res, dict) and res.get("results"):
            for r in res.get("results", []):
                resp = r.get("response", {})
                result_obj = resp.get("result", {})
                if "affected_row_count" in result_obj:
                    rows_affected += result_obj.get("affected_row_count", 0)
        elif isinstance(res, dict) and "rows_affected" in res:
            rows_affected = res.get("rows_affected", 0)

        if rows_affected > 0:
            log.info("Garbage Collection: Cleared payloads for %d old rejected jobs.", rows_affected)
        return rows_affected
    except Exception as exc:
        log.error("Database garbage collection failed: %s", exc)
        return 0
    finally:
        if close_client and client and client.session:
            await client.session.close()


def run_garbage_collection() -> int:
    """Synchronous adapter for run_garbage_collection_async."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(run_garbage_collection_async())).result()
    else:
        return asyncio.run(run_garbage_collection_async())


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "init"

    if cmd == "init":
        asyncio.run(init_db())
        print("Initialized Turso Database schema and indexes successfully.")
    elif cmd == "stats":
        s = asyncio.run(get_stats())
        print(json.dumps(s, indent=2))
    else:
        print("Usage: python database.py [init|stats]")
        sys.exit(1)
