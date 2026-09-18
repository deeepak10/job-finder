"""Scraper package entry point.

collect_all() is what Phase 5's main.py will call. It runs the four
sources, merges them, and applies the cross-run dedup gate once, so the
rest of the pipeline only ever sees jobs it has never processed.
"""

from __future__ import annotations

import asyncio
import logging

from scrapers import adzuna, google_jobs, linkedin, naukri, wellfound, workday
from scrapers.base import JobResult, dedupe, keep_unseen

log = logging.getLogger(__name__)

__all__ = ["collect_all", "collect_all_async", "JobResult", "adzuna", "google_jobs", "linkedin", "naukri", "wellfound", "workday"]


async def collect_all_async(
    include_naukri: bool = True,
    include_google_jobs: bool = True,
    include_wellfound: bool = True,
    include_linkedin: bool = True,
    include_adzuna: bool = True,
    include_workday: bool = True,
    apply_db_dedup: bool = False,
) -> list[JobResult]:
    """Asynchronously collect jobs from all scrapers concurrently."""
    tasks = []
    if include_naukri:
        tasks.append(("naukri", naukri.scrape_async()))
    if include_google_jobs:
        tasks.append(("google_jobs", google_jobs.scrape_async()))
    if include_wellfound:
        tasks.append(("wellfound", wellfound.scrape_async()))
    if include_linkedin:
        tasks.append(("linkedin", linkedin.scrape_async()))
    if include_adzuna:
        tasks.append(("adzuna", adzuna.scrape_async()))
    if include_workday:
        tasks.append(("workday", workday.scrape_async()))

    if not tasks:
        log.info("no scrapers enabled")
        return []

    jobs: list[JobResult] = []
    results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
    for (name, _), res in zip(tasks, results):
        if isinstance(res, Exception):
            log.warning("Scraper task '%s' encountered an exception, bypassed safely: %s", name, res)
        elif isinstance(res, list):
            for item in res:
                if isinstance(item, JobResult):
                    jobs.append(item)
                elif isinstance(item, dict):
                    jobs.append(JobResult(
                        title=item.get("title", ""),
                        company=item.get("company", ""),
                        url=item.get("url", ""),
                        platform=item.get("source", item.get("platform", name)),
                        location=item.get("location", ""),
                        description=item.get("description", ""),
                        raw=item,
                    ))
            log.info("%s scraper collected %d job(s)", name, len(res))

    log.info("collected %d raw job(s)", len(jobs))
    fresh = dedupe(jobs)
    if apply_db_dedup:
        fresh = keep_unseen(fresh)
        log.info("%d new job(s) after database dedup", len(fresh))
    else:
        log.info("%d unique raw job(s) after in-memory dedup", len(fresh))
    return fresh


def collect_all(
    include_naukri: bool = True,
    include_google_jobs: bool = True,
    include_wellfound: bool = True,
    include_linkedin: bool = True,
    include_adzuna: bool = True,
    include_workday: bool = True,
    apply_db_dedup: bool = False,
) -> list[JobResult]:
    """Synchronous adapter for collect_all_async."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(
                collect_all_async(include_naukri, include_google_jobs, include_wellfound, include_linkedin, include_adzuna, include_workday, apply_db_dedup)
            )).result()
    else:
        return asyncio.run(
            collect_all_async(include_naukri, include_google_jobs, include_wellfound, include_linkedin, include_adzuna, include_workday, apply_db_dedup)
        )
