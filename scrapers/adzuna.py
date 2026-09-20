import asyncio
import logging
import os
import aiohttp

logger = logging.getLogger(__name__)


async def scrape_async():
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")

    if not app_id or not app_key:
        logger.warning("Adzuna credentials not found in environment — skipping scraper.")
        return []

    url = (
        "https://api.adzuna.com/v1/api/jobs/in/search/1"
        f"?app_id={app_id}&app_key={app_key}&what=embedded%20firmware&results_per_page=25"
    )

    jobs = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status in (503, 429):
                    logger.warning(f"Adzuna API limit reached. Status: {response.status}. Skipping safely.")
                    return []
                if response.status != 200:
                    logger.warning(f"Adzuna API returned status {response.status} (limit or key error) — skipping safely.")
                    return []

                data = await response.json()
                for item in data.get("results", []):
                    title = item.get("title")
                    job_url = item.get("redirect_url")
                    if not title or not job_url:
                        continue

                    company = item.get("company")
                    if isinstance(company, dict):
                        company_name = company.get("display_name", "Unknown")
                    else:
                        company_name = str(company or "Unknown")

                    location = item.get("location")
                    if isinstance(location, dict):
                        location_name = location.get("display_name", "India")
                    else:
                        location_name = str(location or "India")

                    jobs.append({
                        "title": title.strip(),
                        "company": company_name.strip(),
                        "location": location_name.strip(),
                        "url": job_url.strip(),
                        "source": "adzuna"
                    })
    except Exception as e:
        logger.warning(f"Adzuna API fetch failed safely, ignoring limit/network error: {e}")
        return []

    return jobs


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
