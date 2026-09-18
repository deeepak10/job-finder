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
    get_pending_groq_jobs,
    get_stats,
    init_db,
    is_job_seen,
    mark_alert_sent,
    title_company_hash,
    update_job_status,
)
from discord_alerts import send_discord_alert_async
from evaluator import (
    GeminiQuotaExceededError,
    GroqQuotaExceededError,
    JobEvaluation,
    build_job_prompt,
    evaluate_job,
    evaluate_job_consensus,
    evaluate_job_groq,
    evaluate_with_consensus,
    groq_client,
    openrouter_client,
    query_model,
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
        self.pending_groq_swept: int = 0
        self.pending_groq_verified: int = 0
        self.pending_groq_conflicted: int = 0
        self.pending_groq_added: int = 0
        self.rejected: int = 0
        self.strict_evaluated: int = 0
        self.broad_gatekept: int = 0
        self.lock = asyncio.Lock()

    async def inc_seen(self) -> None:
        async with self.lock:
            self.seen_skipped += 1

    async def inc_spam(self) -> None:
        async with self.lock:
            self.spam_discarded += 1

    async def inc_evaluated(
        self,
        is_high_match: bool,
        alerted: bool,
        is_deferred: bool = False,
        is_rejected: bool = False,
        is_strict: bool = False,
        is_broad: bool = False,
    ) -> None:
        async with self.lock:
            self.evaluated += 1
            if is_high_match:
                self.high_matches += 1
            if alerted:
                self.alerts_sent += 1
            if is_deferred:
                self.deferred_added += 1
            if is_rejected:
                self.rejected += 1
            if is_strict:
                self.strict_evaluated += 1
            if is_broad:
                self.broad_gatekept += 1


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
    deferred_jobs_to_sweep: list[dict[str, Any]] = []
    pending_groq_to_sweep: list[dict[str, Any]] = []

    if not args.dry_run and config.TURSO_DATABASE_URL and config.TURSO_AUTH_TOKEN:
        if sweep_deferred:
            try:
                pending_deferred = await get_deferred_jobs(limit=50)
                if pending_deferred:
                    log.info(
                        "Morning run/flag detected: Swept %d deferred job(s) from Turso queue for Gemini executive tie-breaking.",
                        len(pending_deferred),
                    )
                    print(f"[{_ts()}] [Executive Tie-Breaker] Swept {len(pending_deferred)} deferred job(s) for Gemini re-evaluation.")
                    metrics.deferred_swept = len(pending_deferred)
                    deferred_jobs_to_sweep = pending_deferred
            except Exception as exc:
                log.warning("Could not sweep deferred jobs from Turso: %s", exc)

        # Check for broad jobs pending Groq confirmation (Fallback Ambiguity Rule)
        try:
            pending_groq = await get_pending_groq_jobs(limit=50)
            if pending_groq:
                log.info("Swept %d broad job(s) pending Groq verification from Turso queue.", len(pending_groq))
                print(f"[{_ts()}] [Groq Verification Sweep] Swept {len(pending_groq)} job(s) pending Groq confirmation.")
                metrics.pending_groq_swept = len(pending_groq)
                pending_groq_to_sweep = pending_groq
        except Exception as exc:
            log.warning("Could not sweep pending_groq_verification jobs from Turso: %s", exc)

    if not candidate_jobs and not deferred_jobs_to_sweep and not pending_groq_to_sweep:
        log.info("No fresh candidate, deferred, or pending verification jobs to evaluate. Exiting pipeline.")
        return

    # =========================================================================
    # Phase 3: Tri-Model Evaluation Phase (Executive Tie-Breaker)
    # =========================================================================
    t_eval_start = time.perf_counter()
    # Smart Quota Slicing: Process max 6 candidate jobs per run only when fallbacks are not configured
    if not getattr(config, "GROQ_API_KEY", None) and not getattr(config, "OPENROUTER_API_KEY", None):
        candidate_jobs = candidate_jobs[:6]

    print(f"\n[{_ts()}] >>> [Tri-Model Evaluation Phase] START | Evaluating {len(candidate_jobs)} fresh jobs, {len(deferred_jobs_to_sweep)} deferred, {len(pending_groq_to_sweep)} pending Groq")

    semaphore = asyncio.Semaphore(1)
    gemini_quota_exhausted = False

    async with aiohttp.ClientSession() as session:
        # ---------------------------------------------------------------------
        # Phase 3A.1: Deferred Queue Gemini Executive Tie-Breaker
        # ---------------------------------------------------------------------
        if deferred_jobs_to_sweep:
            print(f"[{_ts()}] >>> [Executive Tie-Breaker] Evaluating {len(deferred_jobs_to_sweep)} deferred conflicts exclusively with Gemini")
            for d_idx, d_job in enumerate(deferred_jobs_to_sweep, 1):
                if gemini_quota_exhausted:
                    log.info("Gemini quota exhausted; halting deferred tie-breaker sweep.")
                    break
                d_payload = {
                    "title": d_job.get("title", ""),
                    "company": d_job.get("company", ""),
                    "location": d_job.get("location", ""),
                    "platform": d_job.get("platform", "deferred"),
                    "description": d_job.get("description", ""),
                    "url": d_job.get("url", ""),
                }
                d_id = d_job.get("job_id") or title_company_hash(d_payload["title"], d_payload["company"])
                log.info(
                    "Executive Tie-Breaker %d/%d (Gemini): '%s' @ %s",
                    d_idx, len(deferred_jobs_to_sweep), d_payload["title"], d_payload["company"],
                )
                try:
                    eval_res = await evaluate_job(d_payload, semaphore)
                    if eval_res:
                        is_match = bool(eval_res.is_match)
                        score = eval_res.match_score or 0
                        new_status = "active" if is_match else "rejected"
                        await update_job_status(
                            job_id=d_id,
                            status=new_status,
                            ai_score=score,
                            visa_sponsorship=eval_res.visa_sponsorship,
                        )
                        alert_sent = False
                        if is_match and score >= 70:
                            alert_sent = await send_discord_alert_async(d_payload, eval_res, session=session)
                            if alert_sent:
                                await mark_alert_sent(d_id)
                        await metrics.inc_evaluated(
                            is_high_match=(is_match and score >= 70),
                            alerted=alert_sent,
                            is_rejected=(not is_match),
                            is_strict=True,
                        )
                except GeminiQuotaExceededError:
                    gemini_quota_exhausted = True
                    log.warning("Gemini quota exhausted during executive tie-breaker sweep. Remaining jobs stay deferred.")
                    break
                except Exception as exc:
                    log.error("Error during executive tie-break for '%s': %s", d_payload["title"], exc)

        # ---------------------------------------------------------------------
        # Phase 3A.2: Pending Groq Verification Sweep (Broad Ambiguity Rule)
        # ---------------------------------------------------------------------
        if pending_groq_to_sweep:
            print(f"[{_ts()}] >>> [Groq Verification Sweep] Verifying {len(pending_groq_to_sweep)} broad role(s) with Groq")
            groq_model = getattr(config, "GROQ_ENSEMBLE_MODEL", "openai/gpt-oss-120b")
            for p_idx, p_job in enumerate(pending_groq_to_sweep, 1):
                p_payload = {
                    "title": p_job.get("title", ""),
                    "company": p_job.get("company", ""),
                    "location": p_job.get("location", ""),
                    "platform": p_job.get("platform", "broad"),
                    "description": p_job.get("description", ""),
                    "url": p_job.get("url", ""),
                }
                p_id = p_job.get("job_id") or title_company_hash(p_payload["title"], p_payload["company"])
                p_prompt = build_job_prompt(p_payload)
                log.info("Groq Verification %d/%d: '%s' @ %s", p_idx, len(pending_groq_to_sweep), p_payload["title"], p_payload["company"])
                try:
                    groq_res = await query_model(groq_client, groq_model, p_prompt)
                    if groq_res is not None:
                        if groq_res.get("is_match") is True:
                            # Consensus reached! Both OpenRouter and Groq accept.
                            score = groq_res.get("ai_score") or 75
                            await update_job_status(
                                job_id=p_id,
                                status="active",
                                ai_score=score,
                                visa_sponsorship=groq_res.get("visa_sponsorship"),
                            )
                            eval_obj = JobEvaluation(
                                is_match=True,
                                visa_sponsorship=groq_res.get("visa_sponsorship") or "Not Specified",
                                ai_score=score,
                                status="active",
                            )
                            alert_sent = False
                            if score >= 70:
                                alert_sent = await send_discord_alert_async(p_payload, eval_obj, session=session)
                                if alert_sent:
                                    await mark_alert_sent(p_id)
                            await metrics.inc_evaluated(
                                is_high_match=(score >= 70),
                                alerted=alert_sent,
                                is_broad=True,
                            )
                            metrics.pending_groq_verified += 1
                            log.info("Pending Groq Verification PASSED for '%s' @ %s. Alert dispatched.", p_payload["title"], p_payload["company"])
                        else:
                            # Conflict reached: OpenRouter accepted previously, but Groq rejected.
                            # Park for next-day Gemini sweep
                            await update_job_status(
                                job_id=p_id,
                                status="deferred",
                                ai_score=0,
                                visa_sponsorship="Conflict",
                            )
                            await metrics.inc_evaluated(
                                is_high_match=False,
                                alerted=False,
                                is_deferred=True,
                                is_broad=True,
                            )
                            metrics.pending_groq_conflicted += 1
                            log.info("Pending Groq Verification CONFLICT for '%s' @ %s. Parked for Gemini sweep.", p_payload["title"], p_payload["company"])
                    else:
                        log.warning("Groq gatekeeper still unavailable for '%s'. Retaining pending_groq_verification status.", p_payload["title"])
                except Exception as p_exc:
                    log.error("Error during Groq verification for '%s': %s", p_payload["title"], p_exc)

        # ---------------------------------------------------------------------
        # Phase 3B: Fresh Candidate Jobs Tri-Model Routing
        # ---------------------------------------------------------------------
        total_candidates = len(candidate_jobs)
        for eval_index, job in enumerate(candidate_jobs, 1):
            job_payload = {
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "platform": job.platform,
                "description": job.description,
                "url": job.url,
            }
            job_id = title_company_hash(job.title, job.company)
            prompt = build_job_prompt(job_payload)
            tier = getattr(job, "tier", "strict")

            if tier == "strict":
                # =============================================================
                # ROUTE A: Strict MedTech Roles (Gemini Primary)
                # =============================================================
                log.info(
                    "Evaluating candidate job %d/%d [Route A - Strict]: '%s' @ %s",
                    eval_index, total_candidates, job.title, job.company,
                )
                evaluation = None
                if not gemini_quota_exhausted:
                    try:
                        evaluation = await evaluate_job(job_payload, semaphore)
                    except GeminiQuotaExceededError:
                        gemini_quota_exhausted = True
                        log.warning("Gemini quota exhausted (429). Activating Junior Consensus for strict roles.")

                if evaluation:
                    status_val = "active" if evaluation.is_match else "rejected"
                    is_high_match = bool(evaluation.is_match and (evaluation.match_score or 0) >= 70)
                    rec = {
                        "job_id": job_id,
                        "title": job.title,
                        "company": job.company,
                        "location": job.location,
                        "platform": job.platform,
                        "url": job.url,
                        "description": job.description,
                        "ai_score": evaluation.match_score,
                        "visa_sponsorship": evaluation.visa_sponsorship,
                        "date_found": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "alert_sent": 0,
                        "status": status_val,
                        "tier": "strict",
                    }
                    alert_delivered = False
                    if not args.dry_run:
                        await add_job(rec)
                        if is_high_match:
                            alert_delivered = await send_discord_alert_async(job_payload, evaluation, session=session)
                            if alert_delivered:
                                await mark_alert_sent(job_id)
                    await metrics.inc_evaluated(
                        is_high_match=is_high_match,
                        alerted=alert_delivered,
                        is_rejected=(not evaluation.is_match),
                        is_strict=True,
                    )
                else:
                    # Fallback to Junior Consensus (Groq + OpenRouter)
                    log.info("Route A Fallback: Querying Junior Consensus for '%s' @ %s", job.title, job.company)
                    consensus_eval = await evaluate_job_consensus(job_payload)
                    c_status = consensus_eval.status

                    if c_status == "consensus_passed":
                        # Both ACCEPTED
                        log.info("Junior Consensus PASSED for '%s' @ %s", job.title, job.company)
                        rec = {
                            "job_id": job_id,
                            "title": job.title,
                            "company": job.company,
                            "location": job.location,
                            "platform": job.platform,
                            "url": job.url,
                            "description": job.description,
                            "ai_score": consensus_eval.ai_score or 75,
                            "visa_sponsorship": consensus_eval.visa_sponsorship,
                            "date_found": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "alert_sent": 0,
                            "status": "consensus_passed",
                            "tier": "strict",
                        }
                        alert_delivered = False
                        if not args.dry_run:
                            await add_job(rec)
                            alert_delivered = await send_discord_alert_async(job_payload, consensus_eval, session=session)
                            if alert_delivered:
                                await mark_alert_sent(job_id)
                        await metrics.inc_evaluated(is_high_match=True, alerted=alert_delivered, is_strict=True)

                    elif c_status == "consensus_failed":
                        # Both REJECTED
                        log.info("Junior Consensus REJECTED '%s' @ %s", job.title, job.company)
                        rec = {
                            "job_id": job_id,
                            "title": job.title,
                            "company": job.company,
                            "location": job.location,
                            "platform": job.platform,
                            "url": job.url,
                            "description": job.description,
                            "ai_score": 0,
                            "visa_sponsorship": "Rejected",
                            "date_found": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "alert_sent": 0,
                            "status": "rejected",
                            "tier": "strict",
                        }
                        if not args.dry_run:
                            await add_job(rec)
                        await metrics.inc_evaluated(is_high_match=False, alerted=False, is_rejected=True, is_strict=True)

                    else:
                        # Split decision (conflict) or quota/model failure -> Defer to Gemini tie-breaker
                        log.info("Junior Consensus CONFLICT/DEFERRED for '%s' @ %s. Parked for Gemini tie-breaker.", job.title, job.company)
                        rec = {
                            "job_id": job_id,
                            "title": job.title,
                            "company": job.company,
                            "location": job.location,
                            "platform": job.platform,
                            "url": job.url,
                            "description": job.description,
                            "ai_score": 0,
                            "visa_sponsorship": "Conflict",
                            "date_found": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "alert_sent": 0,
                            "status": "deferred",
                            "tier": "strict",
                        }
                        if not args.dry_run:
                            await add_job(rec)
                        await metrics.inc_evaluated(is_high_match=False, alerted=False, is_deferred=True, is_strict=True)

            else:
                # =============================================================
                # ROUTE B: Broad Net (Groq Gatekeeper & Fallback Ambiguity Rule)
                # =============================================================
                log.info(
                    "Evaluating candidate job %d/%d [Route B - Broad]: '%s' @ %s (Groq Gatekeeper)",
                    eval_index, total_candidates, job.title, job.company,
                )
                groq_res = None
                groq_online = False
                groq_model = getattr(config, "GROQ_ENSEMBLE_MODEL", "openai/gpt-oss-120b")
                try:
                    groq_res = await query_model(groq_client, groq_model, prompt)
                    if groq_res is not None:
                        groq_online = True
                except Exception as exc:
                    log.warning("Groq gatekeeper exception: %s. Entering Fallback Ambiguity mode.", exc)
                    groq_online = False
                    groq_res = None

                if groq_online and groq_res is not None:
                    # ---------------------------------------------------------
                    # 1 & 2: Groq is ONLINE
                    # ---------------------------------------------------------
                    if groq_res.get("is_match") is False:
                        # Groq rejects -> status = 'rejected', persist to Turso
                        log.info("Route B: Groq gatekeeper rejected '%s' @ %s. Saved Gemini quota.", job.title, job.company)
                        rec = {
                            "job_id": job_id,
                            "title": job.title,
                            "company": job.company,
                            "location": job.location,
                            "platform": job.platform,
                            "url": job.url,
                            "description": job.description,
                            "ai_score": groq_res.get("ai_score") or 0,
                            "visa_sponsorship": groq_res.get("visa_sponsorship") or "Rejected",
                            "date_found": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "alert_sent": 0,
                            "status": "rejected",
                            "tier": "broad",
                        }
                        if not args.dry_run:
                            await add_job(rec)
                        await metrics.inc_evaluated(is_high_match=False, alerted=False, is_rejected=True, is_broad=True)
                        continue

                    # Groq accepts -> Query OpenRouter verification
                    log.info("Route B: Groq gatekeeper accepted '%s' @ %s. Verifying with OpenRouter...", job.title, job.company)
                    or_model = getattr(config, "OPENROUTER_MODEL", "deepseek/deepseek-chat")
                    or_res = None
                    try:
                        or_res = await query_model(openrouter_client, or_model, prompt)
                    except Exception as or_exc:
                        log.warning("OpenRouter verification exception: %s", or_exc)
                        or_res = None

                    if or_res and or_res.get("is_match") is True:
                        # Both accept -> Consensus met. status = 'active', send Discord alert
                        avg_score = int(((groq_res.get("ai_score") or 75) + (or_res.get("ai_score") or 75)) / 2)
                        log.info("Route B: Both models approved '%s' @ %s (Score: %d)", job.title, job.company, avg_score)
                        eval_obj = JobEvaluation(
                            is_match=True,
                            visa_sponsorship=groq_res.get("visa_sponsorship") or or_res.get("visa_sponsorship") or "Not Specified",
                            ai_score=avg_score,
                            status="active",
                        )
                        rec = {
                            "job_id": job_id,
                            "title": job.title,
                            "company": job.company,
                            "location": job.location,
                            "platform": job.platform,
                            "url": job.url,
                            "description": job.description,
                            "ai_score": avg_score,
                            "visa_sponsorship": eval_obj.visa_sponsorship,
                            "date_found": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "alert_sent": 0,
                            "status": "active",
                            "tier": "broad",
                        }
                        alert_delivered = False
                        if not args.dry_run:
                            await add_job(rec)
                            if avg_score >= 70:
                                alert_delivered = await send_discord_alert_async(job_payload, eval_obj, session=session)
                                if alert_delivered:
                                    await mark_alert_sent(job_id)
                        await metrics.inc_evaluated(is_high_match=(avg_score >= 70), alerted=alert_delivered, is_broad=True)

                    else:
                        # OpenRouter rejects -> Conflict met. status = 'deferred', persist for Gemini tie-breaker
                        log.info("Route B Split Decision for '%s' @ %s (Groq=True, OR=%s). Deferring to Gemini tie-breaker.",
                                 job.title, job.company, bool(or_res.get("is_match")) if or_res else None)
                        rec = {
                            "job_id": job_id,
                            "title": job.title,
                            "company": job.company,
                            "location": job.location,
                            "platform": job.platform,
                            "url": job.url,
                            "description": job.description,
                            "ai_score": groq_res.get("ai_score") or 0,
                            "visa_sponsorship": groq_res.get("visa_sponsorship") or "Conflict",
                            "date_found": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "alert_sent": 0,
                            "status": "deferred",
                            "tier": "broad",
                        }
                        if not args.dry_run:
                            await add_job(rec)
                        await metrics.inc_evaluated(is_high_match=False, alerted=False, is_deferred=True, is_broad=True)

                else:
                    # ---------------------------------------------------------
                    # 3: Groq encounters API error -> Fallback Ambiguity Rule
                    # ---------------------------------------------------------
                    log.warning("Route B: Groq offline. Fallback to OpenRouter as solo gatekeeper for '%s' @ %s", job.title, job.company)
                    or_model = getattr(config, "OPENROUTER_MODEL", "deepseek/deepseek-chat")
                    or_res = None
                    try:
                        or_res = await query_model(openrouter_client, or_model, prompt)
                    except Exception as or_exc:
                        log.error("Solo OpenRouter gatekeeper failed: %s", or_exc)
                        or_res = None

                    if or_res and or_res.get("is_match") is False:
                        # OpenRouter rejects -> status = 'rejected', persist to Turso
                        log.info("Route B Solo Gatekeeper: OpenRouter rejected '%s' @ %s.", job.title, job.company)
                        rec = {
                            "job_id": job_id,
                            "title": job.title,
                            "company": job.company,
                            "location": job.location,
                            "platform": job.platform,
                            "url": job.url,
                            "description": job.description,
                            "ai_score": or_res.get("ai_score") or 0,
                            "visa_sponsorship": or_res.get("visa_sponsorship") or "Rejected",
                            "date_found": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "alert_sent": 0,
                            "status": "rejected",
                            "tier": "broad",
                        }
                        if not args.dry_run:
                            await add_job(rec)
                        await metrics.inc_evaluated(is_high_match=False, alerted=False, is_rejected=True, is_broad=True)

                    elif or_res and or_res.get("is_match") is True:
                        # OpenRouter accepts -> DO NOT ALERT. Single junior model cannot approve broad roles.
                        # status = 'pending_groq_verification', persist to Turso
                        log.info(
                            "Route B Fallback Ambiguity: OpenRouter accepted '%s' @ %s. Parked as 'pending_groq_verification'.",
                            job.title, job.company,
                        )
                        rec = {
                            "job_id": job_id,
                            "title": job.title,
                            "company": job.company,
                            "location": job.location,
                            "platform": job.platform,
                            "url": job.url,
                            "description": job.description,
                            "ai_score": or_res.get("ai_score") or 70,
                            "visa_sponsorship": or_res.get("visa_sponsorship") or "Pending Verification",
                            "date_found": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "alert_sent": 0,
                            "status": "pending_groq_verification",
                            "tier": "broad",
                        }
                        if not args.dry_run:
                            await add_job(rec)
                        metrics.pending_groq_added += 1
                        await metrics.inc_evaluated(is_high_match=False, alerted=False, is_broad=True)

                    else:
                        # Both models failed or error occurred -> Defer for Gemini executive tie-breaker
                        log.warning("Route B: Both Groq and OpenRouter failed for '%s' @ %s. Deferring.", job.title, job.company)
                        rec = {
                            "job_id": job_id,
                            "title": job.title,
                            "company": job.company,
                            "location": job.location,
                            "platform": job.platform,
                            "url": job.url,
                            "description": job.description,
                            "ai_score": 0,
                            "visa_sponsorship": "Error",
                            "date_found": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "alert_sent": 0,
                            "status": "deferred",
                            "tier": "broad",
                        }
                        if not args.dry_run:
                            await add_job(rec)
                        await metrics.inc_evaluated(is_high_match=False, alerted=False, is_deferred=True, is_broad=True)

    t_eval_elapsed = time.perf_counter() - t_eval_start
    log.info("========== [Tri-Model Evaluation Phase] END (%s) | Duration: %.2fs | Evaluated: %d jobs ==========",
             _ts(), t_eval_elapsed, metrics.evaluated)
    print(f"[{_ts()}] >>> [Tri-Model Evaluation Phase] END | Duration: {t_eval_elapsed:.2f}s | Evaluated: {metrics.evaluated} jobs")

    total_elapsed = time.perf_counter() - pipeline_start
    log.info("Pipeline execution completed in %.2f seconds.", total_elapsed)

    print("\n" + "=" * 60)
    print(" [PIPELINE RUN METRICS SUMMARY]")
    print("=" * 60)
    print(f" * Total Postings Collected : {metrics.scraped}")
    print(f" * Deferred Swept (Morning) : {metrics.deferred_swept}")
    print(f" * Pending Groq Swept       : {metrics.pending_groq_swept}")
    print(f" * Layer 0 Skipped (Seen)   : {metrics.seen_skipped}")
    print(f" * Layer 1 Discarded (Spam) : {metrics.spam_discarded}")
    print(f" * Layer 4 Evaluated (AI)   : {metrics.evaluated}")
    print(f" * Strict Roles Evaluated   : {metrics.strict_evaluated}")
    print(f" * Broad Roles Gatekept     : {metrics.broad_gatekept}")
    print(f" * Rejections (Quota Saved) : {metrics.rejected}")
    print(f" * High Matches (>= 70)     : {metrics.high_matches}")
    print(f" * Discord Alerts Sent      : {metrics.alerts_sent}")
    print(f" * Jobs Deferred To Queue   : {metrics.deferred_added}")
    print(f" * Pending Groq Added       : {metrics.pending_groq_added}")
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

