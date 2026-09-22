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
from filters import extract_anchored_description, generate_desc_hash

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
    match_reason        TEXT DEFAULT '',
    job_category        TEXT DEFAULT 'General',
    desc_hash           TEXT DEFAULT '',
    retry_count         INTEGER DEFAULT 0
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_jobs_url ON job_postings (url);",
    "CREATE INDEX IF NOT EXISTS idx_jobs_date_found ON job_postings (date_found DESC);",
    "CREATE INDEX IF NOT EXISTS idx_jobs_score ON job_postings (ai_score DESC);",
    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON job_postings (status);",
    "CREATE INDEX IF NOT EXISTS idx_jobs_tier ON job_postings (tier);",
    "CREATE INDEX IF NOT EXISTS idx_jobs_category ON job_postings (job_category);",
    "CREATE INDEX IF NOT EXISTS idx_jobs_desc_hash ON job_postings (desc_hash);",
    "CREATE INDEX IF NOT EXISTS idx_jobs_retries ON job_postings (retry_count);",
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

    Creates the `job_postings` table and performance indexes if they do not exist.
    Heavy DDL column migrations are deferred to run_schema_migrations() to prevent
    startup roundtrip latency and Turso rate limit bloat.

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

        # Add retry_count column for the dynamic quota sweep
        try:
            await client.execute_query("ALTER TABLE job_postings ADD COLUMN retry_count INTEGER DEFAULT 0;")
            log.info("Added column 'retry_count' to job_postings.")
        except Exception as e:
            if "duplicate column name" not in str(e).lower() and "already exists" not in str(e).lower():
                pass  # Ignore error if column already exists

        log.info("Turso database initialized successfully.")
    finally:
        if close_client and client.session:
            await client.session.close()


async def run_schema_migrations(client: Optional[AsyncTursoConnection] = None) -> None:
    """Run non-blocking ALTER TABLE and cleanup migrations on demand.

    Separated from init_db() to eliminate 9 sequential DDL round-trips on pipeline boot.

    Args:
        client: Optional shared AsyncTursoConnection.
    """
    close_client = False
    if client is None:
        client = get_turso_client()
        close_client = True

    try:
        log.info("Running Turso schema migrations...")
        for col_def in (
            ("description", "TEXT"),
            ("status", "TEXT DEFAULT 'active'"),
            ("tier", "TEXT DEFAULT 'strict'"),
            ("match_reason", "TEXT DEFAULT ''"),
            ("job_category", "TEXT DEFAULT 'General'"),
            ("desc_hash", "TEXT DEFAULT ''"),
            ("retry_count", "INTEGER DEFAULT 0"),
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

        log.info("Turso schema migrations completed.")
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


# Note: generate_desc_hash and extract_anchored_description are imported from filters
# to ensure semantic section anchoring (e.g. Qualifications/Requirements) across the pipeline.



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
    description: str = "",
    desc_hash: str = "",
    client: Optional[AsyncTursoConnection] = None,
) -> bool:
    """Execute Layer 0 O(1) deduplication check against Turso Cloud.

    Queries the primary key (`job_id`), indexed `url`, and indexed `desc_hash`
    to determine if a candidate job has already been evaluated or stored in prior runs.

    Args:
        url_or_id: Job posting URL or pre-computed unique identifier.
        title: Job posting title (used with company to derive hash).
        company: Hiring organization name.
        description: Raw job description text (used to compute desc_hash).
        desc_hash: Optional pre-computed 16-hex description hash.
        client: Optional shared AsyncTursoConnection instance.

    Returns:
        True if the job exists in the database; False if fresh or on query error (fail-open).
    """
    clean_url = _strip_url(url_or_id) if url_or_id.startswith("http") else ""
    tc_hash = title_company_hash(title, company) if (title and company) else ""
    target_id = url_or_id if not clean_url else tc_hash
    d_hash = desc_hash or (generate_desc_hash(description) if description else "")

    if d_hash:
        query = (
            "SELECT 1 FROM job_postings "
            "WHERE url = ? OR job_id = ? OR job_id = ? OR (desc_hash != '' AND desc_hash = ?) LIMIT 1"
        )
        args = [clean_url or url_or_id, target_id or url_or_id, tc_hash or url_or_id, d_hash]
    else:
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


async def is_job_duplicate(
    job_id: str = "",
    description: str = "",
    url_or_id: str = "",
    title: str = "",
    company: str = "",
    client: Optional[AsyncTursoConnection] = None,
) -> bool:
    """Checks O(1) existence via ID, URL, and Description Hash to catch recycled duplicates."""
    return await is_job_seen(
        url_or_id=url_or_id or job_id,
        title=title,
        company=company,
        description=description,
        client=client,
    )


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
# Dedicated Async Database Writer Queue (Prevents SQLite Lockouts)
# --------------------------------------------------------------------------

class DatabaseWriteOp:
    """Represents an asynchronous database write operation queued for the worker."""
    def __init__(self, op_type: str, data: Any, future: asyncio.Future):
        self.op_type = op_type
        self.data = data
        self.future = future


class AsyncDatabaseWriter:
    """Funnels all asynchronous inserts/updates through a single dedicated worker
    to eliminate concurrent write lockouts on SQLite / Turso."""

    def __init__(self, client: AsyncTursoConnection):
        self.client = client
        self.queue: asyncio.Queue[Optional[DatabaseWriteOp]] = asyncio.Queue()
        self.task: Optional[asyncio.Task] = None
        self._running = False

    def start(self) -> None:
        if not self._running:
            self._running = True
            self.task = asyncio.create_task(self._worker())
            log.info("AsyncDatabaseWriter queue worker started.")

    async def _worker(self) -> None:
        while self._running:
            item = await self.queue.get()
            if item is None:
                self.queue.task_done()
                break
            op_type = item.op_type
            data = item.data
            fut = item.future
            try:
                if op_type == "add_job":
                    res = await _execute_add_job(data, self.client)
                    if not fut.done():
                        fut.set_result(res)
                elif op_type == "update_status":
                    res = await _execute_update_job_status(
                        job_id=data["job_id"],
                        status=data["status"],
                        ai_score=data.get("ai_score"),
                        visa_sponsorship=data.get("visa_sponsorship"),
                        alert_sent=data.get("alert_sent"),
                        tier=data.get("tier"),
                        match_reason=data.get("match_reason"),
                        job_category=data.get("job_category"),
                        client=self.client,
                    )
                    if not fut.done():
                        fut.set_result(res)
                elif op_type == "mark_alert_sent":
                    await _execute_mark_alert_sent(data, self.client)
                    if not fut.done():
                        fut.set_result(True)
                elif op_type == "bulk_park":
                    res = await _execute_bulk_park_jobs(data["jobs"], data["reason"], self.client)
                    if not fut.done():
                        fut.set_result(res)
                else:
                    if not fut.done():
                        fut.set_result(False)
            except Exception as exc:
                log.error("AsyncDatabaseWriter worker failed processing %s: %s", op_type, exc)
                if not fut.done():
                    fut.set_exception(exc)
            finally:
                self.queue.task_done()

    async def submit(self, op_type: str, data: Any) -> Any:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        op = DatabaseWriteOp(op_type, data, fut)
        await self.queue.put(op)
        return await fut

    async def close(self) -> None:
        if self._running:
            self._running = False
            await self.queue.put(None)
            if self.task:
                await self.task
            log.info("AsyncDatabaseWriter queue drained and stopped.")

    stop = close
    drain_and_stop = close


_GLOBAL_DB_WRITER: Optional[AsyncDatabaseWriter] = None


def get_active_db_writer() -> Optional[AsyncDatabaseWriter]:
    """Retrieve current running AsyncDatabaseWriter instance, if any."""
    return _GLOBAL_DB_WRITER


def start_db_writer(client: AsyncTursoConnection) -> AsyncDatabaseWriter:
    """Start the dedicated global database writer worker."""
    global _GLOBAL_DB_WRITER
    if _GLOBAL_DB_WRITER is None or not _GLOBAL_DB_WRITER._running:
        _GLOBAL_DB_WRITER = AsyncDatabaseWriter(client)
        _GLOBAL_DB_WRITER.start()
    return _GLOBAL_DB_WRITER


async def stop_db_writer() -> None:
    """Drain and cleanly shutdown the global database writer worker."""
    global _GLOBAL_DB_WRITER
    if _GLOBAL_DB_WRITER is not None:
        await _GLOBAL_DB_WRITER.close()
        _GLOBAL_DB_WRITER = None


# --------------------------------------------------------------------------
# Insert & Update Operations
# --------------------------------------------------------------------------

async def _execute_add_job(
    job: dict[str, Any],
    client: Optional[AsyncTursoConnection] = None,
) -> bool:
    """Internal implementation to persist an evaluated job posting record into Turso Cloud SQLite."""
    required = ("title", "company", "platform", "url")
    missing = [k for k in required if not job.get(k)]
    if missing:
        raise ValueError(f"Job is missing required field(s): {', '.join(missing)}")

    title = str(job["title"]).strip()
    company = str(job["company"]).strip()
    url = str(job["url"]).strip()
    platform = str(job["platform"]).strip().lower()
    raw_loc = job.get("location")
    location = raw_loc.strip() if isinstance(raw_loc, str) else "Unknown"

    job_id = job.get("job_id") or title_company_hash(title, company)
    raw_desc = job.get("description")
    description = raw_desc.strip() if isinstance(raw_desc, str) else ""
    raw_score = job.get("ai_score")
    ai_score = int(raw_score) if isinstance(raw_score, (int, float)) else (0 if raw_score is not None else None)
    raw_visa = job.get("visa_sponsorship")
    visa_sponsorship = raw_visa.strip() if isinstance(raw_visa, str) else "Unknown"
    date_found = job.get("date_found") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    alert_sent = int(bool(job.get("alert_sent", 0)))
    raw_status = job.get("status")
    status = raw_status.strip() if isinstance(raw_status, str) else "active"
    raw_tier = job.get("tier")
    tier = raw_tier.strip().lower() if isinstance(raw_tier, str) else "strict"
    raw_reason = job.get("match_reason")
    match_reason = raw_reason.strip() if isinstance(raw_reason, str) else ""
    raw_category = job.get("job_category")
    job_category = raw_category.strip() if isinstance(raw_category, str) and raw_category.strip() else "General"
    raw_desc_hash = job.get("desc_hash")
    desc_hash = (
        raw_desc_hash.strip()
        if isinstance(raw_desc_hash, str) and raw_desc_hash.strip()
        else generate_desc_hash(description)
    )

    sql = """
    INSERT OR REPLACE INTO job_postings (
        job_id, title, company, location, platform, url,
        description, ai_score, visa_sponsorship, date_found,
        alert_sent, status, tier, match_reason, job_category,
        desc_hash
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        job_category,
        desc_hash,
    ]

    close_client = False
    if client is None:
        client = get_turso_client()
        close_client = True

    try:
        res = await client.execute_query(sql, args)
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


async def add_job(
    job: dict[str, Any],
    client: Optional[AsyncTursoConnection] = None,
) -> bool:
    """Persist an evaluated job posting record into Turso Cloud SQLite.
    
    Funnels through dedicated AsyncDatabaseWriter queue when active to prevent write locks.
    """
    writer = get_active_db_writer()
    if client is None and writer is not None and writer._running:
        return await writer.submit("add_job", job)
    return await _execute_add_job(job, client)


async def _execute_bulk_park_jobs(
    jobs: list[Any],
    reason: str = "Ensemble circuit breaker triggered",
    client: Optional[AsyncTursoConnection] = None,
) -> int:
    """Bulk park jobs into Turso using a single batched multi-row query."""
    if not jobs:
        return 0

    close_client = False
    if client is None:
        client = get_turso_client()
        close_client = True

    try:
        now_str = datetime.now(timezone.utc).isoformat(timespec="seconds")
        placeholders = []
        args: list[Any] = []
        for j in jobs:
            if isinstance(j, dict):
                title = j.get("title", "")
                company = j.get("company", "")
                loc = j.get("location", "Unknown")
                plat = j.get("platform", "web")
                url = j.get("url", "")
                desc = j.get("description", "")
                tier = j.get("tier", "strict")
                cat = j.get("job_category", "General")
                j_id = j.get("job_id") or title_company_hash(title, company)
            else:
                title = getattr(j, "title", "")
                company = getattr(j, "company", "")
                loc = getattr(j, "location", "Unknown")
                plat = getattr(j, "platform", "web")
                url = getattr(j, "url", "")
                desc = getattr(j, "description", "")
                tier = getattr(j, "tier", "strict")
                cat = getattr(j, "job_category", "General")
                j_id = getattr(j, "job_id", "") or title_company_hash(title, company)

            d_hash = generate_desc_hash(desc)
            placeholders.append("(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
            args.extend([
                j_id,
                title,
                company,
                loc or "Unknown",
                plat,
                url,
                desc,
                0,
                "Unknown",
                now_str,
                0,
                "parked",
                tier,
                reason,
                cat or "General",
                d_hash,
            ])

        sql = f"""
        INSERT OR REPLACE INTO job_postings (
            job_id, title, company, location, platform, url,
            description, ai_score, visa_sponsorship, date_found,
            alert_sent, status, tier, match_reason, job_category,
            desc_hash
        ) VALUES {', '.join(placeholders)}
        """
        await client.execute_query(sql, args)
        log.info("Bulk parked %d job(s) in Turso with status='parked'.", len(jobs))
        return len(jobs)
    except Exception as exc:
        log.error("Failed to bulk park jobs in Turso: %s", exc)
        return 0
    finally:
        if close_client and client.session:
            await client.session.close()


async def bulk_park_jobs(
    jobs: list[Any],
    reason: str = "Ensemble circuit breaker triggered",
    client: Optional[AsyncTursoConnection] = None,
) -> int:
    """Park all provided candidate jobs via a single Turso query, funneling through writer if active."""
    writer = get_active_db_writer()
    if client is None and writer is not None and writer._running:
        return await writer.submit("bulk_park", {"jobs": jobs, "reason": reason})
    return await _execute_bulk_park_jobs(jobs, reason, client)


async def purge_stale_parked_jobs(client: Optional[AsyncTursoConnection] = None) -> int:
    """Purge stale parked/deferred jobs to provide a clean slate for v2.1 pipeline."""
    sql = """
    UPDATE job_postings 
    SET status = 'discarded', match_reason = 'Stale backlog purged (v2.1 clean slate)'
    WHERE status IN ('parked', 'deferred', 'pending_groq_verification')
    """
    close_client = False
    if client is None:
        client = get_turso_client()
        close_client = True

    try:
        res = await client.execute_query(sql)
        log.info("Purged stale parked/deferred jobs in Turso.")
        return 1
    except Exception as exc:
        log.error("Failed to purge stale parked jobs: %s", exc)
        return 0
    finally:
        if close_client and client.session:
            await client.session.close()


async def fix_workday_urls_in_db(client: Optional[AsyncTursoConnection] = None) -> int:
    """Backfills and fixes existing Workday job records that have broken /External/ links."""
    from scrapers.workday import MEDTECH_WORKDAY_TENANTS
    close_client = False
    if client is None:
        client = get_turso_client()
        close_client = True

    fixed_count = 0
    try:
        for tenant in MEDTECH_WORKDAY_TENANTS:
            site_slug = tenant["url"].rstrip("/").split("/")[-1]
            if site_slug == "External":
                continue
            from urllib.parse import urlsplit
            netloc = urlsplit(tenant["url"]).netloc
            sql_domain = f"%{netloc}%"

            # Fix 1: Replace /External/ with real career site slug
            sql1 = """
            UPDATE job_postings
            SET url = replace(url, '/External/', '/' || ? || '/')
            WHERE url LIKE ? AND url LIKE '%/External/%'
            """
            await client.execute_query(sql1, [site_slug, sql_domain])

            # Fix 2: Clean up any doubled /en-US/en-US/
            sql2 = """
            UPDATE job_postings
            SET url = replace(url, '/en-US/en-US/', '/en-US/')
            WHERE url LIKE ? AND url LIKE '%/en-US/en-US/%'
            """
            await client.execute_query(sql2, [sql_domain])
            fixed_count += 1

        log.info("Workday URL backfill completed across tenants.")
        return fixed_count
    except Exception as exc:
        log.error("Failed to backfill Workday URLs: %s", exc)
        return 0
    finally:
        if close_client and client.session:
            await client.session.close()


async def reclassify_jobs_in_db(client: Optional[AsyncTursoConnection] = None) -> int:
    """Backfills and classifies jobs in Turso DB that are currently marked as 'General'."""
    from discord_alerts import classify_domain_fallback
    close_client = False
    if client is None:
        client = get_turso_client()
        close_client = True

    updated_count = 0
    try:
        res = await client.execute_query(
            "SELECT job_id, title, company FROM job_postings WHERE job_category = 'General'"
        )
        rows = parse_turso_rows(res)
        for r in rows:
            jid = r.get("job_id")
            title = r.get("title", "")
            company = r.get("company", "")
            inferred = classify_domain_fallback(title, company)
            if inferred != "General" and jid:
                await client.execute_query(
                    "UPDATE job_postings SET job_category = ? WHERE job_id = ?",
                    [inferred, jid],
                )
                updated_count += 1
        log.info("Reclassified %d jobs in Turso DB to their true domains.", updated_count)
        return updated_count
    except Exception as exc:
        log.error("Failed to reclassify jobs in Turso: %s", exc)
        return 0
    finally:
        if close_client and client.session:
            await client.session.close()


async def increment_job_retry(
    job_id: str,
    client: Optional[AsyncTursoConnection] = None,
) -> bool:
    """Increment retry_count for a job and automatically mark as 'discarded' if retries >= 3."""
    sql = """
    UPDATE job_postings
    SET retry_count = retry_count + 1,
        status = CASE WHEN retry_count + 1 >= 3 THEN 'discarded' ELSE status END,
        match_reason = CASE WHEN retry_count + 1 >= 3 THEN 'Maximum retries (3) exceeded - discarded' ELSE match_reason END
    WHERE job_id = ?
    """
    close_client = False
    if client is None:
        client = get_turso_client()
        close_client = True

    try:
        await client.execute_query(sql, [job_id])
        return True
    except Exception as exc:
        log.warning("Failed to increment retry_count for job '%s': %s", job_id, exc)
        return False
    finally:
        if close_client and client.session:
            await client.session.close()


async def get_deferred_jobs(
    limit: int = 50,
    client: Optional[AsyncTursoConnection] = None,
) -> list[dict[str, Any]]:
    """Retrieve jobs with status 'deferred' or 'parked' with retry_count < 3.

    Args:
        limit: Maximum number of deferred jobs to retrieve (default: 50).
        client: Optional shared AsyncTursoConnection instance.

    Returns:
        List of dictionaries containing deferred job records.
    """
    sql = "SELECT * FROM job_postings WHERE status IN ('deferred', 'parked') AND retry_count < 3 ORDER BY date_found ASC LIMIT ?"
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


async def _execute_update_job_status(
    job_id: str,
    status: str,
    ai_score: Optional[int] = None,
    visa_sponsorship: Optional[str] = None,
    alert_sent: Optional[int] = None,
    tier: Optional[str] = None,
    match_reason: Optional[str] = None,
    job_category: Optional[str] = None,
    client: Optional[AsyncTursoConnection] = None,
) -> bool:
    """Internal query execution to update status, score, and category in Turso."""
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
    if job_category is not None:
        updates.append("job_category = ?")
        args.append(job_category.strip())

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


async def update_job_status(
    job_id: str,
    status: str,
    ai_score: Optional[int] = None,
    visa_sponsorship: Optional[str] = None,
    alert_sent: Optional[int] = None,
    tier: Optional[str] = None,
    match_reason: Optional[str] = None,
    job_category: Optional[str] = None,
    client: Optional[AsyncTursoConnection] = None,
) -> bool:
    """Update status, score, visa sponsorship, alert status, and category for an existing job posting.
    
    Funnels through AsyncDatabaseWriter queue when active.
    """
    writer = get_active_db_writer()
    if client is None and writer is not None and writer._running:
        payload = {
            "job_id": job_id,
            "status": status,
            "ai_score": ai_score,
            "visa_sponsorship": visa_sponsorship,
            "alert_sent": alert_sent,
            "tier": tier,
            "match_reason": match_reason,
            "job_category": job_category,
        }
        return await writer.submit("update_status", payload)
    return await _execute_update_job_status(
        job_id, status, ai_score, visa_sponsorship, alert_sent, tier, match_reason, job_category, client
    )


async def _execute_mark_alert_sent(
    job_id: str,
    client: Optional[AsyncTursoConnection] = None,
) -> None:
    """Internal query execution to mark a job record as alerted."""
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


async def mark_alert_sent(
    job_id: str,
    client: Optional[AsyncTursoConnection] = None,
) -> None:
    """Mark a job record as alerted after Discord notification succeeds."""
    writer = get_active_db_writer()
    if client is None and writer is not None and writer._running:
        await writer.submit("mark_alert_sent", job_id)
        return
    await _execute_mark_alert_sent(job_id, client)


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
