"""Naukri scraper (Playwright, headless).

Naukri renders job cards client-side, so a plain HTTP fetch returns an
empty shell — hence a real browser.

Selector note: Naukri rotates its class names every few months. Every
selector below is a *list* of candidates tried in order, and the parser
logs loudly when none match, so a DOM change shows up as a warning
rather than a silent zero-results run. Verify SELECTORS against the live
page the first time you run this.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Optional
from urllib.parse import quote_plus

import config
from scrapers.base import JobResult, asleep_jitter, clean, valid_only

log = logging.getLogger(__name__)

PLATFORM = "naukri"

SELECTORS = {
    "card": [
        "div.srp-jobtuple-wrapper",
        "article.jobTuple",
        "div.cust-job-tuple",
        "div.tuple",
        "div[data-job-id]",
    ],
    "title": ["a.title", "a.jobTupleHeader-title", "h2 a", "a[class*='title']"],
    "company": ["a.comp-name", "a.subTitle", "span.comp-name", "a[class*='comp-name']", "span[class*='comp-name']"],
    "location": ["span.locWdth", "span.loc-wrap", "li.location", "span[class*='locWdth']", "span[class*='location']"],
    "snippet": ["span.job-desc", "div.job-description", "span.ellipsis.job-desc", "div[class*='job-desc']"],
}


async def _first_text(card, candidates: list[str]) -> str:
    for sel in candidates:
        node = await card.query_selector(sel)
        if node:
            return clean(await node.inner_text())
    return ""


async def _first_href(card, candidates: list[str]) -> str:
    for sel in candidates:
        node = await card.query_selector(sel)
        if node:
            href = await node.get_attribute("href")
            if href:
                return href if href.startswith("http") else config.NAUKRI_BASE + href
    return ""


def build_search_url(slug: str, keyword: str, page: int = 1) -> str:
    url = f"{config.NAUKRI_BASE}/{slug}"
    if page > 1:
        url += f"-{page}"
    return f"{url}?k={quote_plus(keyword)}"


async def _scrape_page(page, url: str, tier: str = "strict") -> list[JobResult]:
    try:
        await page.goto(url, timeout=config.NAUKRI_TIMEOUT_MS, wait_until="domcontentloaded")
        # Human-like delay after page load to bypass WAF behavioral profiling
        await asyncio.sleep(random.uniform(2.5, 5.0))
    except Exception as exc:
        log.warning("naukri: page load timed out (30s) or failed for %s: %s — skipping", url, exc)
        return []

    card_sel: Optional[str] = None
    for sel in SELECTORS["card"]:
        try:
            await page.wait_for_selector(sel, timeout=5_000)
            card_sel = sel
            break
        except Exception:
            continue

    if card_sel is None:
        log.warning("naukri: no job cards rendered at %s (possibly CAPTCHA or stale selectors) — skipping", url)
        return []

    cards = await page.query_selector_all(card_sel)
    # Cap at top 20 latest results per query
    cards = cards[:20]
    jobs: list[JobResult] = []
    for card in cards:
        link = await _first_href(card, SELECTORS["title"])
        jobs.append(
            JobResult(
                title=await _first_text(card, SELECTORS["title"]),
                company=await _first_text(card, SELECTORS["company"]),
                location=await _first_text(card, SELECTORS["location"]),
                description=await _first_text(card, SELECTORS["snippet"]),
                url=link,
                platform=PLATFORM,
                tier=tier,
            )
        )
    log.info("naukri: %d card(s) from %s (tier: %s)", len(jobs), url, tier)
    return jobs


async def _scrape_via_apify(actor_id: str, apify_token: str) -> list[JobResult]:
    """Scrapes Naukri using an Apify Actor with rotating residential proxies."""
    from datetime import timedelta
    from apify_client import ApifyClientAsync

    client = ApifyClientAsync(token=apify_token)
    run_input = {
        "queries": [
            "medical device",
            "biomedical",
            "firmware",
            "Schiller ECG firmware",
            "Schiller Healthcare R&D",
            "BPL Medical R&D",
            "BPL Medical firmware",
        ],
        "keywords": [
            "medical device",
            "biomedical",
            "firmware",
            "Schiller ECG firmware",
            "Schiller Healthcare R&D",
            "BPL Medical R&D",
            "BPL Medical firmware",
        ],
        "location": "India",
        "maxItems": 25,
    }

    try:
        try:
            run = await client.actor(actor_id).call(
                run_input=run_input,
                timeout=timedelta(seconds=60),
                wait_secs=60,
            )
        except TypeError:
            run = await client.actor(actor_id).call(run_input=run_input)
    except Exception as e:
        log.warning("Naukri Apify actor run failed: %s", e)
        return []

    dataset_id = (
        getattr(run, "default_dataset_id", None)
        or getattr(run, "defaultDatasetId", None)
        or (run.get("defaultDatasetId") if isinstance(run, dict) else None)
    )
    if not dataset_id:
        log.warning("Naukri Apify run returned no dataset ID — skipping.")
        return []

    dataset_items = []
    async for item in client.dataset(dataset_id).iterate_items():
        dataset_items.append(item)

    jobs: list[JobResult] = []
    for item in dataset_items:
        title = item.get("title") or item.get("job_title") or item.get("position")
        company = item.get("company") or item.get("companyName") or item.get("company_name")
        url = item.get("url") or item.get("jobUrl") or item.get("job_url")
        location = item.get("location") or "India"
        description = item.get("description") or item.get("job_description") or ""

        if title and url:
            jobs.append(
                JobResult(
                    title=str(title).strip(),
                    company=str(company or "Unknown").strip(),
                    location=str(location).strip(),
                    url=str(url).strip(),
                    platform=PLATFORM,
                    description=str(description).strip(),
                    tier="strict",
                )
            )
    return jobs


async def scrape(headless: bool = True) -> list[JobResult]:
    """Run every configured Naukri search.
    
    Prefers Apify Actor with residential proxy rotation to bypass Akamai/Cloudflare WAFs in CI.
    Falls back gracefully to local headless Playwright when Apify is not configured.
    """
    import os

    actor_id = getattr(config, "NAUKRI_ACTOR_ID", "") or os.getenv("NAUKRI_ACTOR_ID", "")
    apify_token = getattr(config, "APIFY_TOKEN", "") or os.getenv("APIFY_TOKEN", "")

    if actor_id and apify_token:
        log.info("Scraping Naukri via Apify Actor (%s) with residential proxy rotation...", actor_id)
        try:
            return await _scrape_via_apify(actor_id, apify_token)
        except Exception as exc:
            log.warning("Naukri Apify scraper failed safely: %s", exc)
            return []

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        log.warning("Playwright is not installed. Skipping direct Naukri scraping.")
        return []

    results: list[JobResult] = []
    async with async_playwright() as pw:
        proxy_server = os.getenv("NAUKRI_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
        proxy_config = {"server": proxy_server} if proxy_server else None
        browser = await pw.chromium.launch(
            headless=headless,
            proxy=proxy_config,
            args=[
                "--disable-http2",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = await browser.new_context(
            viewport={"width": 1366, "height": 900},
            locale="en-IN",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            extra_http_headers={
                "Accept-Language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
                "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
            },
        )
        page = await context.new_page()

        # Apply stealth evasions to bypass Cloudflare/WAFs
        try:
            from playwright_stealth import Stealth
            await Stealth().apply_stealth_async(page)
            log.debug("naukri: playwright stealth evasions applied.")
        except Exception:
            try:
                from playwright_stealth import stealth_async
                await stealth_async(page)
                log.debug("naukri: playwright stealth_async evasions applied.")
            except Exception as s_exc:
                log.warning("naukri: could not apply stealth evasions: %s", s_exc)

        page.set_default_timeout(config.NAUKRI_TIMEOUT_MS)
        page.set_default_navigation_timeout(config.NAUKRI_TIMEOUT_MS)

        # Images and fonts are pure cost here — blocking them roughly halves
        # page-load time and bandwidth.
        await page.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in ("image", "font", "media")
            else route.continue_(),
        )

        # Partitioned searches: strict MedTech vs broad generic
        searches_with_tier: list[tuple[str, str, str]] = [
            (slug, kw, "strict") for slug, kw in getattr(config, "NAUKRI_STRICT_SEARCHES", [])
        ] + [
            (slug, kw, "broad") for slug, kw in getattr(config, "NAUKRI_BROAD_SEARCHES", [])
        ]
        if not searches_with_tier:
            searches_with_tier = [
                (slug, kw, config.get_query_tier(slug)) for slug, kw in config.NAUKRI_SEARCHES
            ]

        try:
            for slug, keyword, tier in searches_with_tier:
                # Strictly limit to 1 page per query (top 15-20 latest results)
                max_pages = min(getattr(config, "NAUKRI_MAX_PAGES", 1), 1)
                for page_no in range(1, max_pages + 1):
                    url = build_search_url(slug, keyword, page_no)
                    try:
                        query_jobs = await asyncio.wait_for(
                            _scrape_page(page, url, tier=tier),
                            timeout=30.0,
                        )
                        results.extend(query_jobs[:20])
                    except asyncio.TimeoutError:
                        log.warning("naukri: page processing timed out (30s) for %s — skipping", url)
                    except Exception as exc:
                        log.error("naukri: %s failed (%s)", url, exc)
                    await asleep_jitter(config.NAUKRI_MIN_DELAY, config.NAUKRI_MAX_DELAY)
        finally:
            await context.close()
            await browser.close()

    return valid_only(results)


# Alias for unified asynchronous scraper interface across the pipeline
scrape_async = scrape


if __name__ == "__main__":
    import asyncio
    import json

    logging.basicConfig(level=logging.INFO)
    jobs = asyncio.run(scrape())
    print(json.dumps([{"title": j.title, "company": j.company, "url": j.url}
                      for j in jobs], indent=2)[:4000])
