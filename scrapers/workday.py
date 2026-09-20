import asyncio
import logging
from typing import Any
from urllib.parse import urlsplit
import aiohttp

logger = logging.getLogger(__name__)

MEDTECH_WORKDAY_TENANTS = [
    {
        "name": "Medtronic",
        "url": "https://medtronic.wd1.myworkdayjobs.com/wday/cxs/medtronic/MedtronicCareers",
    },
    {
        "name": "Philips",
        "url": "https://philips.wd3.myworkdayjobs.com/wday/cxs/philips/jobs-and-careers",
    },
]


async def scrape_workday_async() -> list[dict[str, Any]]:
    """Execute direct unauthenticated ATS scraping across enterprise MedTech Workday portals.

    Operates in two asynchronous stages:
        1. Stage 1 (Discovery): Sends a POST request to {tenant_url}/jobs with search
           criteria ('firmware') to discover matching openings.
        2. Stage 2 (Detail Hydration): Sends a secondary GET request to {tenant_url}{external_path}
           to retrieve the full job description (jobPostingInfo.jobDescription) required for
           LLM evaluation.

    Returns:
        List of standardized job dictionaries with title, company, location, url,
        description, and source ('Workday ATS').
    """
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    payload = {
        "appliedFacets": {},
        "limit": 20,
        "offset": 0,
        "searchText": "firmware",
    }

    jobs_collected: list[dict[str, Any]] = []

    async with aiohttp.ClientSession() as session:
        for company in MEDTECH_WORKDAY_TENANTS:
            search_url = f"{company['url']}/jobs"
            try:
                # Step 1: Search for jobs
                async with session.post(search_url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Workday search failed for {company['name']}")
                        continue

                    content_type = getattr(response, "headers", {}).get("Content-Type", "") if hasattr(response, "headers") else ""
                    if content_type and "application/json" not in content_type:
                        logger.warning(
                            f"Workday returned non-JSON response (likely maintenance). Status: {response.status}"
                        )
                        continue

                    data = await response.json()

                    for item in data.get("jobPostings", []):
                        external_path = item.get("externalPath", "")
                        if not external_path:
                            continue

                        detail_url = f"{company['url']}{external_path}"

                        # Secondary fetch for the full job description
                        async with session.get(detail_url, headers=headers) as detail_response:
                            description = "Description not available."
                            if detail_response.status == 200:
                                detail_content_type = getattr(detail_response, "headers", {}).get("Content-Type", "") if hasattr(detail_response, "headers") else ""
                                if not detail_content_type or "application/json" in detail_content_type:
                                    detail_data = await detail_response.json()
                                    job_info = detail_data.get("jobPostingInfo", {})
                                    description = job_info.get("jobDescription", description)
                                else:
                                    logger.warning(
                                        f"Workday returned non-JSON detail response (likely maintenance). Status: {detail_response.status}"
                                    )

                        # Construct valid public-facing candidate URL
                        parsed_base = urlsplit(company["url"])
                        base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"
                        if external_path:
                            public_url = f"{base_domain}/en-US/External{external_path}"
                        else:
                            public_url = base_domain

                        # Step 3: Match the exact Turso database schema
                        jobs_collected.append({
                            "title": item.get("title", ""),
                            "company": company["name"],
                            "location": item.get("locationsText", "Not specified"),
                            "url": public_url,
                            "description": description,
                            "source": "Workday ATS",
                            "tier": "strict",
                        })
            except Exception as e:
                logger.error(f"Failed to scrape Workday for {company['name']}: {e}")

    logger.info(f"Workday ATS: Pulled {len(jobs_collected)} direct enterprise roles.")
    return jobs_collected


# Standard pipeline alias
scrape_async = scrape_workday_async


def scrape() -> list[dict[str, Any]]:
    """Synchronous adapter for scrape_workday_async."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(scrape_workday_async())).result()
    return asyncio.run(scrape_workday_async())


def fetch_workday_jobs(api_url: str) -> list[dict[str, Any]]:
    """Synchronous helper with Content-Type validation to prevent HTML maintenance crashes."""
    import requests
    response = requests.get(api_url)
    if "application/json" in response.headers.get("Content-Type", ""):
        return response.json()
    else:
        logger.warning(
            f"Workday returned non-JSON response (likely maintenance). Status: {response.status_code}"
        )
        return []
