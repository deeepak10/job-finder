import asyncio
import logging
import os
from datetime import timedelta

logger = logging.getLogger(__name__)


async def _scrape_via_apify(actor_id: str, apify_token: str):
    from apify_client import ApifyClientAsync
    client = ApifyClientAsync(token=apify_token)
    run_input = {
        "queries": "embedded firmware",
        "keywords": "embedded firmware",
        "jobTitle": "embedded firmware",
        "title": "embedded firmware",
        "location": "India",
        "maxItems": 20
    }

    try:
        run = await client.actor(actor_id).call(
            run_input=run_input,
            timeout=timedelta(seconds=90),
            wait_secs=90
        )
    except TypeError:
        run = await client.actor(actor_id).call(run_input=run_input)

    dataset_id = getattr(run, "default_dataset_id", None) or getattr(run, "defaultDatasetId", None) or (run.get("defaultDatasetId") if isinstance(run, dict) else None)
    if not dataset_id:
        logger.warning("LinkedIn Apify run returned no dataset ID — skipping.")
        return []

    dataset_items = []
    async for item in client.dataset(dataset_id).iterate_items():
        dataset_items.append(item)

    jobs = []
    for item in dataset_items:
        title = item.get("title") or item.get("job_title") or item.get("position")
        company = item.get("company") or item.get("companyName") or item.get("company_name")
        url = item.get("url") or item.get("jobUrl") or item.get("job_url")
        location = item.get("location") or "Worldwide"

        if title and url:
            jobs.append({
                "title": str(title).strip(),
                "company": str(company or "Unknown").strip(),
                "location": str(location).strip(),
                "url": str(url).strip(),
                "source": "linkedin_apify"
            })
    return jobs


def _run_jobspy_sync():
    from jobspy import scrape_jobs
    jobs_df = scrape_jobs(
        site_name=["linkedin"],
        search_term="embedded firmware medical device",
        location="Worldwide",  # Upgraded to Worldwide
        results_wanted=20,
        hours_old=24,
    )
    if jobs_df is None or jobs_df.empty:
        return []
    return jobs_df.to_dict(orient="records")


async def scrape_async():
    actor_id = os.getenv("LINKEDIN_ACTOR_ID")
    apify_token = os.getenv("APIFY_TOKEN")

    # Mode 1: Apify Residential Proxy (preferred if configured)
    if actor_id and apify_token:
        try:
            logger.info(f"Scraping LinkedIn via Apify Actor ({actor_id})...")
            return await _scrape_via_apify(actor_id, apify_token)
        except Exception as e:
            logger.warning(f"LinkedIn Apify scraper failed safely (ignoring quota/limit): {e}")
            return []

    # Mode 2: Free Guest Scraper (JobSpy) with Fail-Open IP Block Shield
    try:
        logger.info("Scraping LinkedIn via JobSpy (Guest Mode)...")
        raw_jobs = await asyncio.to_thread(_run_jobspy_sync)
        jobs = []
        for item in raw_jobs:
            title = item.get("title")
            url = item.get("job_url") or item.get("url")
            if not title or not url:
                continue
            jobs.append({
                "title": str(title).strip(),
                "company": str(item.get("company") or "Unknown").strip(),
                "location": str(item.get("location") or "Worldwide").strip(),
                "url": str(url).strip(),
                "source": "linkedin_jobspy"
            })
        return jobs
    except Exception as e:
        logger.warning(f"LinkedIn IP block or rate limit detected safely, ignoring and proceeding: {e}")
        return []


def scrape():
    """Synchronous wrapper for scrape_async."""
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
