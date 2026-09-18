"""API Quota Guardian & Schedule Section Manager.

Manages API budgets (SerpApi, Apify) so free-tier monthly limits are never exceeded.
Divides the 24-hour day into 3 distinct sections (Morning, Afternoon, Evening)
and allocates search quotas and location subsets accordingly.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import requests

import config

log = logging.getLogger("quota_manager")
logger = log


def get_active_locations() -> list[str]:
    """Splits the SERPAPI_LOCATIONS master pool into safe, timeout-proof chunks.
    Rotates dynamically based on the current hour (e.g., Morning, Afternoon, Evening runs).
    """
    import config

    all_locations = getattr(config, "SERPAPI_LOCATIONS", ["India"])
    if not all_locations:
        return ["India"]

    # Target 3 scheduled runs per day
    try:
        utc_hour = datetime.now(timezone.utc).hour
    except Exception:
        utc_hour = datetime.utcnow().hour
    run_index = utc_hour % 3

    # Split locations into 3 even chunks
    chunk_size = max(1, len(all_locations) // 3 + (len(all_locations) % 3 > 0))
    chunks = [all_locations[i:i + chunk_size] for i in range(0, len(all_locations), chunk_size)]

    active_locations = chunks[run_index % len(chunks)]
    logger.info("Loaded SerpApi location chunk for current run: %s", active_locations)

    return active_locations


# Default daily limits if quota endpoint cannot be reached
DEFAULT_SERPAPI_MONTHLY = 250
SECTIONS_PER_DAY = 3

# Location rotation mapping across the 3 daily sections
SECTION_LOCATIONS = {
    0: [  # Section 1: Morning (00:00 - 08:00 UTC / 05:30 - 13:30 IST)
        "Bengaluru, Karnataka, India",
        "Hyderabad, Telangana, India",
        "Germany",
    ],
    1: [  # Section 2: Afternoon (08:00 - 16:00 UTC / 13:30 - 21:30 IST)
        "Pune, Maharashtra, India",
        "Chennai, Tamil Nadu, India",
        "United Kingdom",
    ],
    2: [  # Section 3: Evening (16:00 - 24:00 UTC / 21:30 - 05:30 IST)
        "Singapore",
        "Ireland",
        "Netherlands",
    ],
}


def get_current_section() -> int:
    """Return 0, 1, or 2 based on current UTC hour."""
    utc_hour = datetime.now(timezone.utc).hour
    if utc_hour < 8:
        return 0
    elif utc_hour < 16:
        return 1
    return 2


def get_serpapi_account_info() -> dict[str, Any]:
    """Query SerpApi account status and remaining monthly searches."""
    if not config.SERPAPI_API_KEY:
        return {"searches_left": 0, "total_limit": 0, "error": "No API key"}

    try:
        url = f"https://serpapi.com/account?api_key={config.SERPAPI_API_KEY}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "searches_left": data.get("total_searches_left", data.get("plan_searches_left", 0)),
                "total_limit": data.get("searches_per_month", DEFAULT_SERPAPI_MONTHLY),
                "used_this_month": data.get("this_month_usage", 0),
                "plan_name": data.get("plan_name", "Free"),
                "renewal_date": data.get("plan_renewal_date", ""),
            }
        log.warning("serpapi account check returned HTTP %d", resp.status_code)
    except Exception as exc:
        log.warning("serpapi account check failed: %s", exc)

    return {"searches_left": DEFAULT_SERPAPI_MONTHLY, "total_limit": DEFAULT_SERPAPI_MONTHLY}


def calculate_section_search_budget() -> tuple[int, list[str]]:
    """Determine how many SerpApi searches to execute in this section.

    Guarantees the monthly free quota (e.g. 250/month) is never exceeded by:
      1. Checking remaining searches in the billing period.
      2. Dividing remaining searches over remaining days.
      3. Allocating 1/3 of the daily budget to this section.
    """
    info = get_serpapi_account_info()
    searches_left = info.get("searches_left", DEFAULT_SERPAPI_MONTHLY)

    # If critical threshold reached, stop searches to avoid overage
    if searches_left <= 5:
        log.warning(
            "SerpApi monthly quota critical: only %d search(es) remaining. Pausing searches.",
            searches_left,
        )
        return 0, []

    # Calculate days remaining until renewal or standard 30-day window
    days_remaining = 30
    renewal_str = info.get("renewal_date", "")
    if renewal_str:
        try:
            renewal_date = datetime.strptime(renewal_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            delta_days = (renewal_date - datetime.now(timezone.utc)).days
            if delta_days > 0:
                days_remaining = delta_days
        except Exception:
            pass

    daily_budget = max(1, searches_left // days_remaining)
    section_budget = max(1, daily_budget // SECTIONS_PER_DAY)

    section_id = get_current_section()
    planned_locations = SECTION_LOCATIONS.get(section_id, SECTION_LOCATIONS[0])

    # Cap locations to the allowed budget for this section
    target_locations = planned_locations[:section_budget]

    log.info(
        "Section %d/3 active | SerpApi Quota: %d/%d remaining | Budget this section: %d search(es)",
        section_id + 1,
        searches_left,
        info.get("total_limit", DEFAULT_SERPAPI_MONTHLY),
        len(target_locations),
    )
    return len(target_locations), target_locations
