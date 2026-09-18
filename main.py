"""
Master Orchestrator — Unified Autonomous AI Job Pipeline (v2).

Pipeline Execution Workflow:
  1. Initialize Turso Cloud SQLite database schema (idempotent).
  2. Scrape jobs concurrently across Naukri, SerpApi (Google Jobs), and Apify (Wellfound)
     keeping scraping engines 100% intact.
  3. Layer 0 (O(1) Deduplication): Query Turso DB by URL and hash(title + company).
  4. Layer 1 (Regex Spam Filter): Discard medical coding, sales, field maintenance, etc.
  5. Layer 2 (Token Optimization) & Layer 4 (Async LLM Evaluation): Clean with BeautifulSoup,
     truncate to 3,500 chars, and evaluate with Gemini API under asyncio.Semaphore(1) sequentially.
  6. Persistence: Write passing evaluated records to Turso Cloud DB.
  7. Layer 5 (Non-Blocking Discord Alerts): Dispatch alerts via aiohttp only for
     is_match == True and match_score >= 70.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any

# Ensure line-buffering so logs stream in real-time in GitHub Actions
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import aiohttp

import config
from database import (
    add_job,
    get_deferred_jobs,
    get_stats,
    init_db,
    is_job_seen,
    mark_alert_sent,
    title_company_hash,
    update_job_status,
)
from evaluator import (
    GeminiQuotaExceededError,
    GroqQuotaExceededError,
    JobEvaluation,
    evaluate_job,
    evaluate_job_consensus,
    evaluate_job_groq,
    evaluate_with_consensus,
)
from filters import clean_html_text, is_spam_title
from scrapers import collect_all, collect_all_async
from scrapers.base import JobResult

log = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Autonomous AI Job Pipeline (v2) - Semantic Job Evaluator"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run scraping, filtering, and LLM evaluation without persisting to Turso or sending Discord alerts.",
    )
    parser.add_argument(
        "--re-eval-deferred",
        action="store_true",
        help="Force sweep and re-evaluation of deferred queue from Turso Cloud Database.",
    )
    parser.add_argument(
        "--no-naukri",
        action="store_true",
        help="Skip the Naukri Playwright scraper.",
    )
    parser.add_argument(
        "--no-google-jobs",
        action="store_true",
        help="Skip SerpApi Google Jobs engine scraper (LinkedIn & Indeed).",
    )
    parser.add_argument(
        "--no-wellfound",
        action="store_true",
        help="Skip Wellfound startup jobs scraper.",
    )
    parser.add_argument(
        "--no-linkedin",
        action="store_true",
        help="Skip LinkedIn scraper.",
    )
    parser.add_argument(
        "--no-adzuna",
        action="store_true",
        help="Skip Adzuna scraper.",
    )
    parser.add_argument(
        "--no-workday",
        action="store_true",
        help="Skip Workday direct enterprise ATS scraper.",
    )
    parser.add_argument(
        "--force-quota-scrapers",
        action="store_true",
        help="Force execution of quota-heavy scrapers (SerpApi, LinkedIn, Wellfound) regardless of run hour.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Query Turso Cloud Database metrics and exit.",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate environment configuration and print status with masked credentials.",
    )
    parser.add_argument(
        "--test-alert",
        action="store_true",
        help="Dispatch sample high-priority Discord alert to verify webhook integration.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )
    return parser.parse_args()


def print_config_check() -> None:
    """Print safe configuration health report without leaking raw credentials."""
    status = config.validate_configuration()
    print("=" * 60)
    print(" [SECURITY & CONFIGURATION AUDIT REPORT]")
    print("=" * 60)
    print(f" * Turso Database URL : {config.mask_secret(config.TURSO_DATABASE_URL, 14, 10)}")
    print(f" * Turso Auth Token    : {config.mask_secret(config.TURSO_AUTH_TOKEN, 6, 6)}")
    print(f" * Gemini API Key      : {config.mask_secret(config.GEMINI_API_KEY, 6, 6)}")
    print(f" * Gemini Model        : {config.GEMINI_MODEL}")
    print(f" * Groq API Key        : {config.mask_secret(config.GROQ_API_KEY, 4, 4)}")
    print(f" * Groq Fallback Models: llama-3.3-70b-versatile -> llama-3.1-8b-instant")
    print(f" * Groq Ensemble Model : {config.GROQ_ENSEMBLE_MODEL}")
    print(f" * OpenRouter API Key  : {config.mask_secret(config.OPENROUTER_API_KEY, 8, 4)}")
    print(f" * OpenRouter Model    : {config.OPENROUTER_MODEL}")
    print(f" * Discord Webhook     : {config.mask_secret(config.DISCORD_WEBHOOK_URL, 35, 6)}")
    print(f" * SerpApi Key         : {config.mask_secret(config.SERPAPI_API_KEY, 4, 4)}")
    print(f" * Apify Token         : {config.mask_secret(config.APIFY_TOKEN, 8, 4)}")
    print("-" * 60)
    if status["issues"]:
        print(" [!] Configuration Issues Found:")
        for issue in status["issues"]:
            print(f"     - {issue}")
    else:
        print(" [OK] All critical credentials and security constraints are satisfied.")
    print("=" * 60)



async def run_test_alert() -> None:
    """Send a realistic test alert to Discord demonstrating the v2 embed layout."""
    test_job = {
        "title": "Biomedical Systems & Embedded Software Engineer",
        "company": "GE HealthCare",
        "location": "Bengaluru, Karnataka, India",
        "platform": "linkedin",
        "url": "https://www.linkedin.com/jobs/view/test-biomedical-systems",
    }
    test_eval = JobEvaluation(
        is_match=True,
        visa_sponsorship="Local Indian role (Direct Employment / No Sponsorship Needed)",
    )
    log.info("Dispatching test alert to Discord...")
    success = await send_discord_alert_async(test_job, test_eval)
    if success:
        log.info("Test alert successfully delivered to Discord!")
    else:
        log.error("Failed to deliver test alert. Check DISCORD_WEBHOOK_URL.")


class PipelineMetrics:
    def __init__(self) -> None:
        self.scraped: int = 0
        self.seen_skipped: int = 0
        self.spam_discarded: int = 0
        self.deferred_swept: int = 0
        self.evaluated: int = 0
        self.high_matches: int = 0
        self.alerts_sent: int = 0
        self.deferred_added: int = 0
        self.lock = asyncio.Lock()

    async def inc_seen(self) -> None:
        async with self.lock:
            self.seen_skipped += 1

    async def inc_spam(self) -> None:
        async with self.lock:
            self.spam_discarded += 1

    async def inc_evaluated(self, is_high_match: bool, alerted: bool, is_deferred: bool = False) -> None:
        async with self.lock:
            self.evaluated += 1
            if is_high_match:
                self.high_matches += 1
            if alerted:
                self.alerts_sent += 1
            if is_deferred:
                self.deferred_added += 1


async def process_job(
    job: JobResult,
    semaphore: asyncio.Semaphore,
    dry_run: bool,
    session: aiohttp.ClientSession,
    metrics: Optional[PipelineMetrics] = None,
) -> None:
    """Process a single scraped job through Layers 0, 1, 2, 4, persistence, and Discord alerting."""
    # --- Layer 0: O(1) Deduplication against Turso ---
    if await is_job_seen(url_or_id=job.url, title=job.title, company=job.company):
        log.info("Layer 0: Skipped already-seen job '%s' @ %s", job.title, job.company)
        if metrics:
            await metrics.inc_seen()
        return

    # --- Layer 1: Regex Spam Filter ---
    is_spam, spam_reason = is_spam_title(job.title)
    if is_spam:
        log.info("Layer 1: Discarded spam '%s' @ %s (%s)", job.title, job.company, spam_reason)
        if metrics:
            await metrics.inc_spam()
        return

    # --- Layer 2 & 4: Token Optimization + Concurrent Gemini LLM Evaluation ---
    job_payload = {
        "title": job.title,
        "company": job.company,
        "location": job.location,
        "platform": job.platform,
        "description": job.description,
        "url": job.url,
    }
    try:
        evaluation = await evaluate_job(job_payload, semaphore)
    except GeminiQuotaExceededError:
        if getattr(config, "OPENROUTER_API_KEY", None) and getattr(config, "GROQ_API_KEY", None):
            evaluation = await evaluate_job_consensus(job_payload)
        elif getattr(config, "GROQ_API_KEY", None):
            try:
                from groq import AsyncGroq
                groq_client = AsyncGroq(api_key=config.GROQ_API_KEY)
                evaluation = await evaluate_job_groq(job_payload, groq_client)
            except Exception as e:
                log.warning("Groq fallback in process_job failed: %s", e)
                evaluation = None
        else:
            evaluation = None

    if not evaluation:
        log.warning("Evaluation failed or skipped for '%s' @ %s", job.title, job.company)
        return

    status_val = getattr(evaluation, "status", "active")
    is_high_match = bool(evaluation.is_match)
    job_id = title_company_hash(job.title, job.company)
    record = {
        "job_id": job_id,
        "title": job.title,
        "company": job.company,
        "location": job.location,
        "platform": job.platform,
        "url": job.url,
        "description": job.description,
        "ai_score": getattr(evaluation, "match_score", None),
        "visa_sponsorship": evaluation.visa_sponsorship,
        "date_found": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "alert_sent": 0,
        "status": status_val,
    }

    if dry_run:
        log.info(
            "[DRY RUN] Evaluated '%s' @ %s -> Match: %s, Status: %s",
            job.title,
            job.company,
            evaluation.is_match,
            status_val,
        )
        if metrics:
            await metrics.inc_evaluated(
                is_high_match=(is_high_match and status_val != "deferred"),
                alerted=False,
                is_deferred=(status_val == "deferred"),
            )
        return

    # --- Persistence to Turso Cloud ---
    saved = await add_job(record)
    if saved:
        log.info("Saved evaluated job to Turso: '%s' @ %s (status: %s)", job.title, job.company, status_val)

    # --- Layer 5: Non-Blocking Discord Notification Engine ---
    alert_delivered = False
    if is_high_match and status_val != "deferred":
        alert_delivered = await send_discord_alert_async(job_payload, evaluation, session=session)
        if alert_delivered:
            await mark_alert_sent(job_id)

    if metrics:
        await metrics.inc_evaluated(
            is_high_match=(is_high_match and status_val != "deferred"),
            alerted=alert_delivered,
            is_deferred=(status_val == "deferred"),
        )


async def run_scrapers_concurrently(
    args: Optional[argparse.Namespace] = None,
    current_hour_utc: Optional[int] = None,
) -> list[JobResult]:
    """Execute all scrapers concurrently with quota time-gating and complete task isolation.

    Zero-cost and unlimited scrapers (Naukri, Adzuna, and direct Workday ATS) execute
    on every run. Quota-capped scrapers (SerpApi Google Jobs, LinkedIn Apify, and
    Wellfound Apify) execute only during the morning window (current_hour_utc <= 4,
    corresponding to <= 10:00 AM IST / 02:00 UTC) to strictly preserve monthly limits.

    Args:
        args: Optional CLI arguments namespace containing scraper toggle flags.
        current_hour_utc: Optional override for current UTC hour (useful for unit testing).
            Defaults to datetime.now(timezone.utc).hour.

    Returns:
        List of collected and flattened JobResult instances across all active scrapers.
    """
    from scrapers import adzuna, google_jobs, linkedin, naukri, wellfound, workday

    if current_hour_utc is None:
        current_hour_utc = datetime.now(timezone.utc).hour

    # GitHub Actions run in UTC. 02:00 UTC corresponds to 07:30 AM IST.
    is_morning_run = current_hour_utc <= 4  # Triggers for cron jobs before 10:00 AM IST
    force_quota = getattr(args, "force_quota_scrapers", False) if args else False

    tasks = []

    # Free, unlimited scrapers run every time
    if getattr(config, "NAUKRI_ENABLED", True) and not (args and getattr(args, "no_naukri", False)):
        tasks.append(naukri.scrape_async())
    if getattr(config, "ADZUNA_ENABLED", True) and not (args and getattr(args, "no_adzuna", False)):
        tasks.append(adzuna.scrape_async())
    if getattr(config, "WORKDAY_ENABLED", True) and not (args and getattr(args, "no_workday", False)):
        tasks.append(workday.scrape_async())

    # Time-gated quota-heavy scrapers
    if is_morning_run or force_quota:
        log.info("Morning run detected: Executing SerpApi, LinkedIn, and Wellfound...")
        if getattr(config, "GOOGLE_JOBS_ENABLED", True) and not (args and getattr(args, "no_google_jobs", False)):
            tasks.append(google_jobs.scrape_async())
        if getattr(config, "LINKEDIN_ENABLED", True) and not (args and getattr(args, "no_linkedin", False)):
            tasks.append(linkedin.scrape_async())
        if getattr(config, "WELLFOUND_ENABLED", True) and not (args and getattr(args, "no_wellfound", False)):
            tasks.append(wellfound.scrape_async())
    else:
        log.info("Off-peak run: Bypassing quota-heavy scrapers.")

    log.info("Running %d scraper task(s) concurrently with complete isolation...", len(tasks))
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_jobs: list[JobResult] = []
    for res in results:
        if isinstance(res, list):
            for item in res:
                if isinstance(item, JobResult):
                    all_jobs.append(item)
                elif isinstance(item, dict):
                    all_jobs.append(JobResult(
                        title=item.get("title", ""),
                        company=item.get("company", ""),
                        url=item.get("url", ""),
                        platform=item.get("source", item.get("platform", "web")),
                        location=item.get("location", ""),
                        description=item.get("description", ""),
                        raw=item,
                    ))
        else:
            log.warning("Scraper task encountered an exception, bypassed safely: %s", res)

    return all_jobs


async def run_pipeline_async(args: argparse.Namespace) -> None:
    """Execute the end-to-end autonomous job discovery and evaluation pipeline.

    Workflow Stages:
        1. Scraping Phase: Concurrent unauthenticated/API scraping with quota time-gating.
        2. Deduplication & Filtering Phase: Layer 0 O(1) Turso indexed URL/hash dedup,
           followed by Layer 1 regex spam filtering and Layer 2 text cleaning.
        3. Multi-Provider LLM Evaluation Phase: Primary evaluation via Gemini 3.6 Flash
           with 4.5s pacing, failing over seamlessly to Groq Llama-3.1-8B (with 6s pacing)
           upon encountering HTTP 429 RESOURCE_EXHAUSTED.
        4. Persistence & Notifications: Saves match records to Turso Cloud SQLite and
           dispatches rich embeds to Discord for high-match roles (is_match=True).

    Args:
        args: Parsed command-line argument namespace.
    """
    pipeline_start = time.perf_counter()
    metrics = PipelineMetrics()

    def _ts() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    log.info("Starting Unified Autonomous AI Job Pipeline (v2)%s at %s...",
             " [DRY RUN]" if args.dry_run else "", _ts())

    # Ensure Turso schema is ready
    if not args.dry_run:
        await init_db()

    # =========================================================================
    # Phase 1: [Scraping Phase]
    # =========================================================================
    t_scrape_start = time.perf_counter()
    log.info("========== [Scraping Phase] START (%s) ==========", _ts())
    print(f"\n[{_ts()}] >>> [Scraping Phase] START")

    from scrapers.base import dedupe

    all_jobs = await run_scrapers_concurrently(args=args)
    scraped_jobs = dedupe(all_jobs)
    metrics.scraped = len(scraped_jobs)
    t_scrape_elapsed = time.perf_counter() - t_scrape_start
    log.info("========== [Scraping Phase] END (%s) | Duration: %.2fs | Collected: %d jobs ==========",
             _ts(), t_scrape_elapsed, metrics.scraped)
    print(f"[{_ts()}] >>> [Scraping Phase] END | Duration: {t_scrape_elapsed:.2f}s | Collected: {metrics.scraped} jobs")

    if not scraped_jobs:
        log.info("No scraped jobs to process. Exiting pipeline.")
        return

    # =========================================================================
    # Phase 2: [Database Deduplication Phase]
    # =========================================================================
    t_dedup_start = time.perf_counter()
    log.info("========== [Database Deduplication Phase] START (%s) ==========", _ts())
    print(f"\n[{_ts()}] >>> [Database Deduplication Phase] START")

    # Step A: Layer 1 Regex Spam Filter (immediate CPU regex, zero network cost)
    non_spam_jobs: list[JobResult] = []
    for job in scraped_jobs:
        is_spam, spam_reason = is_spam_title(job.title)
        if is_spam:
            log.info("Layer 1: Discarded spam '%s' @ %s (%s)", job.title, job.company, spam_reason)
            await metrics.inc_spam()
        else:
            non_spam_jobs.append(job)

    # Step B: Layer 0 O(1) Turso Database Deduplication with connection reuse
    candidate_jobs: list[JobResult] = []
    turso_client = None
    if not args.dry_run and config.TURSO_DATABASE_URL and config.TURSO_AUTH_TOKEN:
        try:
            from database import get_turso_client
            turso_client = get_turso_client()
        except Exception as exc:
            log.warning("Could not initialize pooled Turso client: %s", exc)

    try:
        async def check_seen(job: JobResult) -> Optional[JobResult]:
            seen = await is_job_seen(url_or_id=job.url, title=job.title, company=job.company, client=turso_client)
            if seen:
                log.info("Layer 0: Skipped already-seen job '%s' @ %s", job.title, job.company)
                await metrics.inc_seen()
                return None
            return job

        # Check all non-spam candidates concurrently
        results = await asyncio.gather(*(check_seen(j) for j in non_spam_jobs))
        candidate_jobs = [j for j in results if j is not None]
    finally:
        if turso_client and turso_client.session:
            await turso_client.session.close()

    t_dedup_elapsed = time.perf_counter() - t_dedup_start
    log.info("========== [Database Deduplication Phase] END (%s) | Duration: %.2fs | Filtered: %d -> %d fresh jobs ==========",
             _ts(), t_dedup_elapsed, len(scraped_jobs), len(candidate_jobs))
    print(f"[{_ts()}] >>> [Database Deduplication Phase] END | Duration: {t_dedup_elapsed:.2f}s | Fresh: {len(candidate_jobs)} jobs")

    if not candidate_jobs:
        log.info("No fresh candidate jobs to evaluate after deduplication.")

    # Check for deferred jobs to sweep and re-evaluate during morning run or explicit flag
    current_hour_utc = datetime.now(timezone.utc).hour
    is_morning_run = current_hour_utc <= 4
    sweep_deferred = is_morning_run or getattr(args, "re_eval_deferred", False)

    if sweep_deferred and not args.dry_run and config.TURSO_DATABASE_URL and config.TURSO_AUTH_TOKEN:
        try:
            pending_deferred = await get_deferred_jobs(limit=50)
            if pending_deferred:
                log.info(
                    "Morning run/flag detected: Swept %d deferred job(s) from Turso queue for re-evaluation.",
                    len(pending_deferred),
                )
                print(f"[{_ts()}] [Queue] Swept {len(pending_deferred)} deferred job(s) from Turso for morning re-evaluation.")
                metrics.deferred_swept = len(pending_deferred)
                seen_ids = {title_company_hash(j.title, j.company) for j in candidate_jobs}
                for d_job in pending_deferred:
                    d_hash = d_job.get("job_id") or title_company_hash(d_job.get("title", ""), d_job.get("company", ""))
                    if d_hash not in seen_ids:
                        candidate_jobs.append(JobResult(
                            title=d_job.get("title", ""),
                            company=d_job.get("company", ""),
                            url=d_job.get("url", ""),
                            platform=d_job.get("platform", "deferred"),
                            location=d_job.get("location", ""),
                            description=d_job.get("description", ""),
                        ))
                        seen_ids.add(d_hash)
        except Exception as exc:
            log.warning("Could not sweep deferred jobs from Turso: %s", exc)

    if not candidate_jobs:
        log.info("No fresh candidate or deferred jobs to evaluate. Exiting pipeline.")
        return

    # =========================================================================
    # Phase 3: [Gemini Evaluation Phase]
    # =========================================================================
    t_eval_start = time.perf_counter()
    # Smart Quota Slicing: Process max 6 candidate jobs per run only when fallbacks are not configured
    if not getattr(config, "GROQ_API_KEY", None) and not getattr(config, "OPENROUTER_API_KEY", None):
        candidate_jobs = candidate_jobs[:6]

    print(f"\n[{_ts()}] >>> [Evaluation Phase] START | Evaluating {len(candidate_jobs)} jobs")

    # Concurrency control: asyncio.Semaphore(1) for LLM evaluation (sequential processing)
    semaphore = asyncio.Semaphore(1)

    async def evaluate_and_persist(
        job: JobResult,
        session: aiohttp.ClientSession,
        use_fallback: bool = False,
        use_consensus: bool = False,
        groq_client: Any = None,
    ) -> None:
        job_payload = {
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "platform": job.platform,
            "description": job.description,
            "url": job.url,
        }
        if not use_fallback:
            evaluation = await evaluate_job(job_payload, semaphore)
        elif use_consensus:
            # Dual-model consensus fallback (Groq + OpenRouter)
            evaluation = await evaluate_job_consensus(job_payload)
        else:
            evaluation = await evaluate_job_groq(job_payload, groq_client)

        if not evaluation:
            log.warning("Evaluation failed, timed out, or skipped for '%s' @ %s", job.title, job.company)
            return

        status_val = getattr(evaluation, "status", "active")
        is_high_match = bool(evaluation.is_match)
        job_id = title_company_hash(job.title, job.company)
        record = {
            "job_id": job_id,
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "platform": job.platform,
            "url": job.url,
            "description": job.description,
            "ai_score": getattr(evaluation, "match_score", None),
            "visa_sponsorship": evaluation.visa_sponsorship,
            "date_found": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "alert_sent": 0,
            "status": status_val,
        }

        if args.dry_run:
            log.info(
                "[DRY RUN] Evaluated '%s' @ %s -> Match: %s, Status: %s",
                job.title,
                job.company,
                evaluation.is_match,
                status_val,
            )
            await metrics.inc_evaluated(
                is_high_match=(is_high_match and status_val != "deferred"),
                alerted=False,
                is_deferred=(status_val == "deferred"),
            )
            return

        # Persistence to Turso Cloud (persists active, consensus_passed, and deferred records)
        saved = await add_job(record)
        if saved:
            log.info("Saved evaluated job to Turso: '%s' @ %s (status: %s)", job.title, job.company, status_val)

        # Layer 5: Non-Blocking Discord Notification Engine (Only alert on confirmed matches)
        alert_delivered = False
        if is_high_match and status_val != "deferred":
            alert_delivered = await send_discord_alert_async(job_payload, evaluation, session=session)
            if alert_delivered:
                await mark_alert_sent(job_id)

        await metrics.inc_evaluated(
            is_high_match=(is_high_match and status_val != "deferred"),
            alerted=alert_delivered,
            is_deferred=(status_val == "deferred"),
        )

    use_fallback = False
    use_consensus = bool(getattr(config, "OPENROUTER_API_KEY", None))
    groq_client = None
    queue = list(candidate_jobs)
    total_candidates = len(candidate_jobs)
    eval_index = 0

    async with aiohttp.ClientSession() as session:
        while queue:
            job = queue.pop(0)
            eval_index += 1
            if not use_fallback:
                provider_tag = "Gemini"
            elif use_consensus:
                provider_tag = "Ensemble Consensus (Groq + OpenRouter)"
            else:
                provider_tag = "Groq (Fallback Rotation)"

            log.info(
                "Evaluating candidate job %d/%d with %s: '%s' @ %s",
                eval_index,
                total_candidates,
                provider_tag,
                job.title,
                job.company,
            )
            try:
                if not use_fallback:
                    await evaluate_and_persist(job, session, use_fallback=False)
                else:
                    if not use_consensus and not groq_client:
                        if getattr(config, "GROQ_API_KEY", None):
                            try:
                                from groq import AsyncGroq
                                groq_client = AsyncGroq(api_key=config.GROQ_API_KEY)
                            except ImportError:
                                log.error("groq package is not installed. Cannot use fallback. Aborting.")
                                break
                        else:
                            log.error("Groq API key missing. Cannot use fallback. Aborting.")
                            break

                    await evaluate_and_persist(
                        job,
                        session,
                        use_fallback=True,
                        use_consensus=use_consensus,
                        groq_client=groq_client,
                    )

            except Exception as exc:
                error_msg = str(exc)
                code = getattr(exc, "status_code", getattr(exc, "code", None))
                is_quota = (
                    isinstance(exc, (GeminiQuotaExceededError, GroqQuotaExceededError))
                    or code == 429
                    or "429" in error_msg
                    or "RESOURCE_EXHAUSTED" in error_msg
                )

                if is_quota:
                    if not use_fallback:
                        log.warning(
                            "Gemini API quota exhausted (429 RESOURCE_EXHAUSTED): %s. Activating secondary fallback...",
                            exc,
                        )
                        print(f"[{_ts()}] [!] Gemini API quota exhausted (429). Activating secondary fallback tier...")
                        use_fallback = True
                        eval_index -= 1
                        queue.insert(0, job)  # Retry current job with fallback
                        continue
                    else:
                        log.warning("Secondary fallback quota/rate limit reached: %s. Deferring remaining queue to Turso.", exc)
                        print(f"[{_ts()}] [!] Secondary fallback limit reached. Deferring remaining queue to Turso.")
                        # Securely persist remaining jobs as deferred in Turso
                        if not args.dry_run:
                            for remaining_job in [job] + queue:
                                rec = {
                                    "job_id": title_company_hash(remaining_job.title, remaining_job.company),
                                    "title": remaining_job.title,
                                    "company": remaining_job.company,
                                    "location": remaining_job.location,
                                    "platform": remaining_job.platform,
                                    "url": remaining_job.url,
                                    "description": remaining_job.description,
                                    "ai_score": 0,
                                    "visa_sponsorship": "Unknown",
                                    "date_found": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                    "alert_sent": 0,
                                    "status": "deferred",
                                }
                                await add_job(rec)
                                metrics.deferred_added += 1
                        break
                else:
                    log.error("Unexpected evaluation error for '%s' @ %s: %s", job.title, job.company, exc)
                    continue

    t_eval_elapsed = time.perf_counter() - t_eval_start
    log.info("========== [Gemini Evaluation Phase] END (%s) | Duration: %.2fs | Evaluated: %d jobs ==========",
             _ts(), t_eval_elapsed, metrics.evaluated)
    print(f"[{_ts()}] >>> [Gemini Evaluation Phase] END | Duration: {t_eval_elapsed:.2f}s | Evaluated: {metrics.evaluated} jobs")

    total_elapsed = time.perf_counter() - pipeline_start
    log.info("Pipeline execution completed in %.2f seconds.", total_elapsed)

    print("\n" + "=" * 60)
    print(" [PIPELINE RUN METRICS SUMMARY]")
    print("=" * 60)
    print(f" * Total Postings Collected : {metrics.scraped}")
    print(f" * Deferred Swept (Morning) : {metrics.deferred_swept}")
    print(f" * Layer 0 Skipped (Seen)   : {metrics.seen_skipped}")
    print(f" * Layer 1 Discarded (Spam) : {metrics.spam_discarded}")
    print(f" * Layer 4 Evaluated (AI)   : {metrics.evaluated}")
    print(f" * High Matches (>= 70)     : {metrics.high_matches}")
    print(f" * Discord Alerts Sent      : {metrics.alerts_sent}")
    print(f" * Jobs Deferred To Queue   : {metrics.deferred_added}")
    print(f" * Execution Duration       : {total_elapsed:.2f}s")
    print("=" * 60 + "\n")



def main() -> None:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s : %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.check_config:
        print_config_check()
        sys.exit(0)

    if args.stats:
        stats_data = asyncio.run(get_stats())
        print(json.dumps(stats_data, indent=2))
        sys.exit(0)

    if args.test_alert:
        asyncio.run(run_test_alert())
        sys.exit(0)

    try:
        asyncio.run(run_pipeline_async(args))
    except KeyboardInterrupt:
        log.info("Pipeline execution cancelled by user. Exiting cleanly.")
        sys.exit(130)


if __name__ == "__main__":
    main()

