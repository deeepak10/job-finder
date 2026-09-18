"""LinkedIn + Indeed via SerpApi's Google Jobs engine.

Google Jobs aggregates many boards, so results are filtered down to the
two we want using apply_options. One SerpApi call per location, which is
the unit that costs you credits — keep the location lists in config tight.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import aiohttp

import config
from scrapers.base import JobResult, clean, dedupe, sleep_jitter, valid_only

log = logging.getLogger(__name__)

PAGES_PER_LOCATION = 1          # Hardcoded max 1 page = 10 results per SerpApi credit
MAX_RESULTS_PER_QUERY = 20     # Strict cap at top 20 latest results
SERP_MIN_DELAY, SERP_MAX_DELAY = 1.0, 2.0


class MissingCredentials(RuntimeError):
    pass


def _pick_apply_link(result: dict[str, Any]) -> tuple[str, str]:
    """Return (url, platform) for the first LinkedIn/Indeed apply option.

    Google Jobs lists several apply routes per posting; we want the one on
    a board we actually target, not whichever happens to be first.
    """
    options = result.get("apply_options") or []
    for opt in options:
        title = (opt.get("title") or "").lower()
        link = opt.get("link") or ""
        for board in config.ALLOWED_JOB_BOARDS:
            if board in title or board in link.lower():
                return link, board

    # Older responses use related_links instead.
    for rel in result.get("related_links") or []:
        link = rel.get("link", "")
        for board in config.ALLOWED_JOB_BOARDS:
            if board in link.lower():
                return link, board

    return "", ""


def _to_job(result: dict[str, Any]) -> Optional[JobResult]:
    url, platform = _pick_apply_link(result)
    if not url:
        return None      # posting exists only on a board we don't track
    return JobResult(
        title=result.get("title", ""),
        company=result.get("company_name", ""),
        location=result.get("location", ""),
        description=clean(result.get("description", "")),
        url=url,
        platform=platform,
        posted=(result.get("detected_extensions") or {}).get("posted_at", ""),
        raw=result,
    )


async def _search_async(
    session: aiohttp.ClientSession,
    query: str,
    location: str,
    page: int = 0,
) -> list[dict[str, Any]]:
    """Native asynchronous SerpApi query using aiohttp with 30s timeout enforcement."""
    if not config.SERPAPI_API_KEY:
        raise MissingCredentials("SERPAPI_API_KEY is not set")

    params: dict[str, Any] = {
        "engine": "google_jobs",
        "q": query,
        "location": location,
        "hl": "en",
        "api_key": config.SERPAPI_API_KEY,
    }
    if page:
        params["start"] = str(page * 10)

    try:
        async def _fetch():
            async with session.get("https://serpapi.com/search", params=params) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    log.error("serpapi(%s, query='%s'): HTTP %d: %s", location, query, resp.status, text[:200])
                    return {}
                return await resp.json()

        data = await asyncio.wait_for(_fetch(), timeout=30.0)
    except asyncio.TimeoutError:
        log.warning("serpapi(%s, query='%s'): HTTP request timed out after 30s", location, query)
        return []
    except Exception as exc:
        log.error("serpapi(%s, query='%s'): request failed: %s", location, query, exc)
        return []

    if "error" in data:
        log.error("serpapi(%s): %s", location, data["error"])
        return []
    return data.get("jobs_results", []) or []


def _search(query: str, location: str, page: int = 0) -> list[dict[str, Any]]:
    """Synchronous bridge for callers and tests."""
    if not config.SERPAPI_API_KEY:
        raise MissingCredentials("SERPAPI_API_KEY is not set")

    async def _runner():
        async with aiohttp.ClientSession() as session:
            return await _search_async(session, query, location, page)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(_runner())).result()
    else:
        return asyncio.run(_runner())


_DEFAULT_SEARCH_FN = _search


async def _execute_search(
    session: aiohttp.ClientSession,
    query: str,
    location: str,
    page: int = 0,
) -> list[dict[str, Any]]:
    """Execute search respecting monkeypatched _search in tests or native async in production."""
    if _search is not _DEFAULT_SEARCH_FN:
        res = _search(query, location, page)
        if asyncio.iscoroutine(res):
            return await res
        return res
    return await _search_async(session, query, location, page)


async def _scrape_locations_async(
    locations: list[str],
    label: str,
    session: Optional[aiohttp.ClientSession] = None,
) -> list[JobResult]:
    jobs: list[JobResult] = []
    queries = getattr(config, "GOOGLE_JOBS_QUERIES", getattr(config, "TARGET_QUERIES", [config.GOOGLE_JOBS_QUERY]))
    max_pages = min(getattr(config, "SERPAPI_MAX_PAGES", 1), 1)
    max_results = getattr(config, "SERPAPI_MAX_RESULTS_PER_QUERY", MAX_RESULTS_PER_QUERY)

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True

    try:
        tasks = [(q, loc) for q in queries for loc in locations]

        async def _fetch_one(query: str, location: str) -> list[JobResult]:
            query_jobs: list[JobResult] = []
            for page in range(max_pages):
                try:
                    raw = await asyncio.wait_for(
                        _execute_search(session, query, location, page),
                        timeout=30.0,
                    )
                except MissingCredentials:
                    raise
                except asyncio.TimeoutError:
                    log.warning("serpapi %s/%s query '%s' timed out (30s)", label, location, query)
                    break
                except Exception as exc:
                    log.warning("serpapi %s/%s query '%s' failed: %s", label, location, query, exc)
                    break
                if not raw:
                    break
                for r in raw[:max_results]:
                    job = _to_job(r)
                    if job:
                        query_jobs.append(job)
            return query_jobs[:max_results]

        results = await asyncio.gather(*[_fetch_one(q, loc) for q, loc in tasks], return_exceptions=True)
        for (q, loc), res in zip(tasks, results):
            if isinstance(res, MissingCredentials):
                raise res
            elif isinstance(res, Exception):
                log.error("serpapi %s/%s ['%s'] failed: %s", label, loc, q, res)
            elif isinstance(res, list):
                jobs.extend(res)
                log.info("serpapi %s/%s ['%s']: found %d job(s)", label, loc, q, len(res))
    finally:
        if close_session:
            await session.close()

    return dedupe(valid_only(jobs))


def _scrape_locations(locations: list[str], label: str) -> list[JobResult]:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(_scrape_locations_async(locations, label))).result()
    else:
        return asyncio.run(_scrape_locations_async(locations, label))


def scrape_india() -> list[JobResult]:
    return _scrape_locations(config.INDIA_LOCATIONS, "india")


def scrape_global() -> list[JobResult]:
    """Europe, the UK and Singapore — the roles that need the visa filter."""
    return _scrape_locations(config.GLOBAL_LOCATIONS, "global")


async def scrape_async(
    locations: Optional[list[str]] = None,
    session: Optional[aiohttp.ClientSession] = None,
) -> list[JobResult]:
    """Scrape Google Jobs (LinkedIn & Indeed) budgeted by the section quota."""
    if locations is None:
        try:
            from quota_manager import get_active_locations
            locations = get_active_locations()
        except Exception as exc:
            log.warning("quota_manager get_active_locations unavailable (%s); using default locations", exc)
            locations = getattr(config, "SERPAPI_LOCATIONS", ["India"])

    if not locations:
        log.info("serpapi: no locations scheduled for this section")
        return []

    return await _scrape_locations_async(locations, "section_budget", session=session)


def scrape(locations: Optional[list[str]] = None) -> list[JobResult]:
    """Synchronous adapter for scrape_async."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(scrape_async(locations))).result()
    else:
        return asyncio.run(scrape_async(locations))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for job in scrape():
        print(f"[{job.platform}] {job.title} — {job.company} ({job.location})")
