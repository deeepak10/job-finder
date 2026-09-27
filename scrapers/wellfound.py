"""Wellfound (health-tech startups) via an Apify actor.

Wellfound has no public jobs API, so this goes through a third-party
actor. Actor output schemas differ between providers and change without
notice, so _normalize() accepts several common field names rather than
assuming one shape — set WELLFOUND_ACTOR_ID in .env and check the first
run's log line to confirm the mapping caught your actor's fields.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import config
from scrapers.base import JobResult, clean, valid_only

log = logging.getLogger(__name__)

PLATFORM = "wellfound"

# Candidate keys, in priority order, per output field.
FIELD_MAP = {
    "title": ("job_title", "title", "jobTitle", "role", "name"),
    "company": ("company_name", "company", "companyName", "startupName", "organization"),
    "location": ("location", "locationNames", "locations", "city"),
    "url": ("URL", "url", "jobUrl", "link", "applyUrl", "jobListingUrl"),
    "description": ("description", "jobDescription", "descriptionText", "summary"),
    "job_id": ("id", "jobId", "slug"),
}


class MissingCredentials(RuntimeError):
    pass


def _first(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value if v)
        if isinstance(value, dict):
            value = value.get("name") or value.get("title") or ""
        if value:
            return clean(str(value))
    return ""


def _normalize(item: dict[str, Any]) -> Optional[JobResult]:
    if not isinstance(item, dict):
        return None

    title = item.get("job_title") or item.get("title") or _first(item, FIELD_MAP["title"])
    company = item.get("company_name") or item.get("company") or _first(item, FIELD_MAP["company"])
    url = item.get("URL") or item.get("url") or _first(item, FIELD_MAP["url"])
    location = item.get("location") or _first(item, FIELD_MAP["location"]) or "Not specified"

    if not title or not company or not url:
        return None

    title = clean(str(title))
    company = clean(str(company))
    if isinstance(location, (list, tuple)):
        location = ", ".join(str(v) for v in location if v)
    location = clean(str(location)) or "Not specified"

    url = clean(str(url))
    if url and not url.startswith("http"):
        url = "https://wellfound.com" + ("" if url.startswith("/") else "/") + url

    native_id = _first(item, FIELD_MAP["job_id"])
    description = _first(item, FIELD_MAP["description"])

    return JobResult(
        title=title,
        company=company,
        location=location,
        description=description,
        url=url,
        platform=PLATFORM,
        # Prefer Wellfound's own id: startup listing URLs get rewritten when
        # a role is edited, which would otherwise look like a brand new job.
        job_id=f"wf-{native_id}" if native_id else "",
        raw=item,
    )


def _run_actor() -> list[dict[str, Any]]:
    if not config.APIFY_TOKEN:
        raise MissingCredentials("APIFY_TOKEN is not set")
    if not config.WELLFOUND_ACTOR_ID:
        raise MissingCredentials("WELLFOUND_ACTOR_ID is not set")

    from datetime import timedelta
    from apify_client import ApifyClient

    client = ApifyClient(config.APIFY_TOKEN)
    run_input = {
        "search": "Health Tech",
        "locations": ["India", "United States", "United Kingdom", "Germany", "Netherlands", "Singapore", "Ireland"],
        "industryTags": getattr(config, "WELLFOUND_INDUSTRY_TAGS", ["Health Tech", "Medical Devices"]),
        "industries": getattr(config, "WELLFOUND_INDUSTRY_TAGS", ["Health Tech", "Medical Devices"]),
        "keywords": config.WELLFOUND_KEYWORDS,
        "skills": config.WELLFOUND_SKILLS,
        "maxItems": min(getattr(config, "WELLFOUND_MAX_RESULTS", 20), 20),
    }
    try:
        run = client.actor(config.WELLFOUND_ACTOR_ID).call(
            run_input=run_input,
            wait_duration=timedelta(seconds=90),
            run_timeout=timedelta(seconds=90),
        )
    except TypeError:
        # Fallback for test mocks that do not take timeout arguments
        run = client.actor(config.WELLFOUND_ACTOR_ID).call(run_input=run_input)

    if not run:
        log.error("wellfound: actor run returned no dataset")
        return []

    try:
        dataset_id = run.default_dataset_id
    except AttributeError:
        dataset_id = run.get("defaultDatasetId") if isinstance(run, dict) else getattr(run, "defaultDatasetId", None)

    if not dataset_id:
        log.error("wellfound: actor run returned no dataset")
        return []
    items = list(client.dataset(dataset_id).iterate_items())
    return items[:20]


def matches_focus(job: JobResult) -> bool:
    """Actor-side filters are unreliable, so re-check on our side:
    a health-tech signal plus at least one of our skills."""
    text = f"{job.title} {job.company} {job.description}".lower()
    industry_tags = getattr(config, "WELLFOUND_INDUSTRY_TAGS", ["Health Tech", "Medical Devices"])
    health = any(k.lower() in text for k in industry_tags) or any(
        w in text for w in ("health", "medical", "clinical", "biotech", "patient", "diagnostic")
    )
    skill = (
        any(s.lower() in text for s in config.WELLFOUND_SKILLS)
        or any(w in text for w in ("firmware", "embedded", "iot", "machine learning", "signal processing", "r&d"))
        or any(k.lower() in text for k in config.WELLFOUND_KEYWORDS)
    )
    return health and skill


async def scrape_async() -> list[JobResult]:
    """Asynchronous scrape with strict 90s timeout on actor execution."""
    try:
        items = await asyncio.wait_for(
            asyncio.to_thread(_run_actor),
            timeout=90.0,
        )
    except MissingCredentials as exc:
        log.warning("wellfound: skipped (%s)", exc)
        return []
    except asyncio.TimeoutError:
        log.warning("wellfound: actor run timed out (90s) — skipping")
        return []
    except Exception as exc:
        log.error("wellfound: actor failed (%s)", exc)
        return []

    # Strict cap at top 20 latest results per run
    items = items[:20]
    if items:
        log.info("wellfound: first item keys = %s", sorted(items[0].keys()))

    jobs: list[JobResult] = []
    for item in items:
        job = _normalize(item)
        if job is None:
            continue
        jobs.append(job)

    jobs = valid_only(jobs)
    focused = [j for j in jobs if matches_focus(j)]
    log.info("wellfound: %d item(s) -> %d valid -> %d on-focus",
             len(items), len(jobs), len(focused))
    return focused


def scrape() -> list[JobResult]:
    """Synchronous adapter for scrape_async."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(scrape_async())).result()
    else:
        return asyncio.run(scrape_async())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for job in scrape():
        print(f"{job.title} — {job.company} ({job.location})")
