import asyncio
import logging
from typing import Any
from urllib.parse import urlsplit
import aiohttp

logger = logging.getLogger(__name__)

# Expanded Workday Tenant URLs for Top MedTech Companies
MEDTECH_WORKDAY_TENANTS = [
    {"name": "Medtronic", "url": "https://medtronic.wd1.myworkdayjobs.com/wday/cxs/medtronic/MedtronicCareers"},
    {"name": "Philips", "url": "https://philips.wd3.myworkdayjobs.com/wday/cxs/philips/jobs-and-careers"},
    {"name": "Stryker", "url": "https://stryker.wd1.myworkdayjobs.com/wday/cxs/stryker/StrykerCareers"},
    {"name": "GE HealthCare", "url": "https://gehc.wd5.myworkdayjobs.com/wday/cxs/gehc/GEHC_ExternalSite"},
    {"name": "Abbott", "url": "https://abbott.wd5.myworkdayjobs.com/wday/cxs/abbott/abbottcareers2"},
    {"name": "Becton Dickinson (BD)", "url": "https://bd.wd1.myworkdayjobs.com/wday/cxs/bd/bdcareers"},
    {"name": "Baxter", "url": "https://baxter.wd1.myworkdayjobs.com/wday/cxs/baxter/baxter"},
    {"name": "Danaher", "url": "https://danaher.wd1.myworkdayjobs.com/wday/cxs/danaher/DanaherCareers"},
    {"name": "Zimmer Biomet", "url": "https://zimmerbiomet.wd1.myworkdayjobs.com/wday/cxs/zimmerbiomet/Zimmer_Biomet_Careers"},
    {"name": "Smith+Nephew", "url": "https://smithnephew.wd5.myworkdayjobs.com/wday/cxs/smithnephew/External"},
]

# Maximum Search Options mapped to the Candidate Profile
SEARCH_KEYWORDS = [
    "firmware",
    "biomedical",
    "embedded",
    "iot",
    "signal processing",
    "telemetry",
    "computer vision",
    "python",
]


async def scrape_workday_async() -> list[dict[str, Any]]:
    """Execute direct unauthenticated ATS scraping across enterprise MedTech Workday portals."""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    jobs_collected: list[dict[str, Any]] = []
    seen_urls = set()  # Prevent duplicate ingestion if multiple keywords match the same job

    async with aiohttp.ClientSession() as session:
        for company in MEDTECH_WORKDAY_TENANTS:
            search_url = f"{company['url']}/jobs"

            for keyword in SEARCH_KEYWORDS:
                payload = {
                    "appliedFacets": {},
                    "limit": 20,
                    "offset": 0,
                    "searchText": keyword,
                }

                try:
                    # Anti-Bot Pacing: Prevent Workday 429 Rate Limits
                    await asyncio.sleep(1.5)

                    async with session.post(search_url, json=payload, headers=headers) as response:
                        if response.status != 200:
                            if response.status == 429:
                                logger.warning(
                                    f"Workday Rate Limit (429) hit for {company['name']}. Skipping remaining keywords."
                                )
                                break  # Break keyword loop, move to next company
                            continue

                        content_type = (
                            getattr(response, "headers", {}).get("Content-Type", "")
                            if hasattr(response, "headers")
                            else ""
                        )
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

                            # Construct valid public-facing candidate URL
                            parsed_base = urlsplit(company["url"])
                            base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"
                            public_url = f"{base_domain}/en-US/External{external_path}"

                            # Prevent hydrating the same job twice if it matches multiple keywords
                            if public_url in seen_urls:
                                continue
                            seen_urls.add(public_url)

                            detail_url = f"{company['url']}{external_path}"

                            # Anti-Bot Pacing for detail hydration
                            await asyncio.sleep(0.5)

                            async with session.get(detail_url, headers=headers) as detail_response:
                                description = "Description not available."
                                if detail_response.status == 200:
                                    detail_content_type = (
                                        getattr(detail_response, "headers", {}).get("Content-Type", "")
                                        if hasattr(detail_response, "headers")
                                        else ""
                                    )
                                    if not detail_content_type or "application/json" in detail_content_type:
                                        detail_data = await detail_response.json()
                                        job_info = detail_data.get("jobPostingInfo", {})
                                        description = job_info.get("jobDescription", description)
                                    else:
                                        logger.warning(
                                            f"Workday returned non-JSON detail response (likely maintenance). Status: {detail_response.status}"
                                        )

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
                    logger.error(f"Failed to scrape Workday for {company['name']} with keyword '{keyword}': {e}")

    logger.info(
        f"Workday ATS: Pulled {len(jobs_collected)} direct enterprise roles across {len(MEDTECH_WORKDAY_TENANTS)} companies."
    )
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
