import asyncio
import logging
from typing import Any
from urllib.parse import urlsplit
import aiohttp
from aiohttp import ClientTimeout

logger = logging.getLogger(__name__)

# Expanded Workday Tenant URLs for Top MedTech Companies
MEDTECH_WORKDAY_TENANTS = [
    {"name": "Medtronic", "slug": "medtronic", "url": "https://medtronic.wd1.myworkdayjobs.com/wday/cxs/medtronic/MedtronicCareers"},
    {"name": "Philips", "slug": "philips", "url": "https://philips.wd3.myworkdayjobs.com/wday/cxs/philips/jobs-and-careers"},
    {"name": "Stryker", "slug": "stryker", "url": "https://stryker.wd1.myworkdayjobs.com/wday/cxs/stryker/StrykerCareers"},
    {"name": "GE HealthCare", "slug": "gehealthcare", "url": "https://gehc.wd5.myworkdayjobs.com/wday/cxs/gehc/GEHC_ExternalSite"},
    {"name": "Getinge (Maquet)", "slug": "getinge", "url": "https://getinge.wd3.myworkdayjobs.com/wday/cxs/getinge/Getinge_Careers"},
    {"name": "Abbott", "slug": "abbott", "url": "https://abbott.wd5.myworkdayjobs.com/wday/cxs/abbott/abbottcareers2"},
    {"name": "Becton Dickinson (BD)", "slug": "bd", "url": "https://bd.wd1.myworkdayjobs.com/wday/cxs/bd/bdcareers"},
    {"name": "Baxter", "slug": "baxter", "url": "https://baxter.wd1.myworkdayjobs.com/wday/cxs/baxter/baxter"},
    {"name": "Danaher", "slug": "danaher", "url": "https://danaher.wd1.myworkdayjobs.com/wday/cxs/danaher/DanaherCareers"},
    {"name": "Zimmer Biomet", "slug": "zimmerbiomet", "url": "https://zimmerbiomet.wd1.myworkdayjobs.com/wday/cxs/zimmerbiomet/Zimmer_Biomet_Careers"},
    {"name": "Smith+Nephew", "slug": "smithnephew", "url": "https://smithnephew.wd5.myworkdayjobs.com/wday/cxs/smithnephew/External"},
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


def build_workday_candidate_url(company_url: str, external_path: str) -> str:
    """Constructs valid public-facing SPA candidate URL with career site slug."""
    parsed_base = urlsplit(company_url)
    base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"
    site_slug = company_url.rstrip("/").split("/")[-1]
    return f"{base_domain}/en-US/{site_slug}{external_path}"


async def _scrape_company_workday(
    session: aiohttp.ClientSession,
    company: dict[str, str],
    semaphore: asyncio.Semaphore,
    seen_urls: set[str],
    headers: dict[str, str],
) -> list[dict[str, Any]]:
    """Scrapes a single Workday enterprise tenant across configured keywords."""
    company_jobs: list[dict[str, Any]] = []
    search_url = f"{company['url']}/jobs"

    async with semaphore:
        for keyword in SEARCH_KEYWORDS:
            payload = {
                "appliedFacets": {},
                "limit": 20,
                "offset": 0,
                "searchText": keyword,
            }

            try:
                # Anti-Bot Pacing per keyword within the company
                await asyncio.sleep(0.3)

                async with session.post(search_url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        if response.status == 429:
                            logger.warning(
                                f"Workday Rate Limit (429) hit for {company['name']}. Skipping remaining keywords."
                            )
                            break
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

                        # Construct valid public-facing candidate URL: https://{tenant}.myworkdayjobs.com/en-US/{site_slug}{external_path}
                        public_url = build_workday_candidate_url(company["url"], external_path)

                        # Prevent hydrating the same job twice if it matches multiple keywords
                        if public_url in seen_urls:
                            continue
                        seen_urls.add(public_url)

                        detail_url = f"{company['url']}{external_path}"
                        description = "Description not available."

                        try:
                            # Anti-Bot Pacing for detail hydration
                            await asyncio.sleep(0.2)
                            async with session.get(detail_url, headers=headers) as detail_response:
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
                        except Exception as det_err:
                            logger.debug("Detail fetch error for %s: %s", detail_url, det_err)

                        company_jobs.append({
                            "title": item.get("title", ""),
                            "company": company["name"],
                            "location": item.get("locationsText", "Not specified"),
                            "url": public_url,
                            "description": description,
                            "source": "Workday ATS",
                            "tier": "strict",
                        })
            except asyncio.TimeoutError:
                logger.warning(f"Workday Timeout: {company['name']} hung on keyword '{keyword}'. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Failed to scrape Workday for {company['name']} with keyword '{keyword}': {e}")

    return company_jobs


async def scrape_workday_async() -> list[dict[str, Any]]:
    """Execute direct unauthenticated ATS scraping across enterprise MedTech Workday portals concurrently."""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    seen_urls: set[str] = set()
    # Strict 10-second network timeout to prevent silent pipeline hangs
    timeout = ClientTimeout(total=10)
    # Concurrency bounded by Semaphore(5)
    semaphore = asyncio.Semaphore(5)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            _scrape_company_workday(session, company, semaphore, seen_urls, headers)
            for company in MEDTECH_WORKDAY_TENANTS
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    jobs_collected: list[dict[str, Any]] = []
    for res in results:
        if isinstance(res, list):
            jobs_collected.extend(res)
        elif isinstance(res, Exception):
            logger.warning("Workday company scrape error: %s", res)

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
