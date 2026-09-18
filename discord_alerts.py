"""
Asynchronous Discord Notification Engine (v2).

Features:
  * Non-blocking webhook delivery via aiohttp
  * Strict gating: Triggers ONLY if is_match == True
  * Dynamic embed coloring:
      - 986895 (Purple) for priority healthcare targets (Philips, GE HealthCare, Stryker, Skanray, etc.)
      - 3447003 (Blue) for other companies
  * Rich Embed layout:
      - Header: 🟩 {job_title} (clickable hyperlink pointing to url)
      - Inline Fields: Company, Location, Platform, Visa Status
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

import aiohttp

import config
from evaluator import JobEvaluation

log = logging.getLogger(__name__)

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Decimal color codes specified in Master Spec v2
COLOR_PURPLE = 986895   # Target healthcare companies
COLOR_BLUE = 3447003    # Standard matching roles


def pick_embed_color(company: str) -> int:
    """Determine embed color based on target healthcare company affiliation."""
    comp_clean = (company or "").lower()
    for target in config.TARGET_HEALTHCARE_COMPANIES:
        if target.lower() in comp_clean:
            return COLOR_PURPLE
    return COLOR_BLUE


def _safe_str(val: Any, max_len: int = 1000, fallback: str = "—") -> str:
    """Sanitize and truncate text to prevent Discord API 400 Bad Request limits."""
    s = str(val or "").strip()
    if not s:
        return fallback
    return s[:max_len].strip()


def build_discord_embed(
    job: dict[str, Any],
    evaluation: JobEvaluation,
) -> dict[str, Any]:
    """Construct the Discord Rich Embed."""
    title = _safe_str(job.get("title"), max_len=240, fallback="Job Posting")
    url = (job.get("url") or "").strip()
    company = _safe_str(job.get("company"), max_len=200, fallback="—")
    location = _safe_str(job.get("location"), max_len=200, fallback="—")
    platform = _safe_str((job.get("platform") or "Web").capitalize(), max_len=50, fallback="Web")
    color = pick_embed_color(company)

    fields = [
        {"name": "🏢 Company", "value": company, "inline": True},
        {"name": "📍 Location", "value": location, "inline": True},
        {"name": "🌐 Platform", "value": platform, "inline": True},
        {
            "name": "🛂 Visa Status",
            "value": _safe_str(evaluation.visa_sponsorship, max_len=200, fallback="Not Specified"),
            "inline": True,
        },
    ]

    embed = {
        "title": f"🟩 {title}",
        "url": url,
        "color": color,
        "fields": fields,
    }

    return embed


async def send_discord_alert_async(
    job: dict[str, Any],
    evaluation: JobEvaluation,
    webhook_url: Optional[str] = None,
    session: Optional[aiohttp.ClientSession] = None,
) -> bool:
    """Send alert to Discord webhook asynchronously.

    Enforces gating: Only dispatches if is_match is True.
    """
    if not evaluation.is_match:
        log.info(
            "Skipping Discord alert for '%s' (is_match=%s)",
            job.get("title"),
            evaluation.is_match,
        )
        return False

    target_url = webhook_url or config.DISCORD_WEBHOOK_URL or DISCORD_WEBHOOK_URL
    if not target_url:
        log.warning("DISCORD_WEBHOOK_URL not set; skipping notification.")
        return False

    # SSRF protection: only allow official Discord webhook endpoints
    if not target_url.startswith(
        ("https://discord.com/api/webhooks/", "https://discordapp.com/api/webhooks/")
    ):
        log.error("Security alert: DISCORD_WEBHOOK_URL must point to official discord.com endpoints.")
        return False

    embed = build_discord_embed(job, evaluation)
    payload = {"embeds": [embed]}

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True

    try:
        async with session.post(target_url, json=payload, timeout=aiohttp.ClientTimeout(total=15, connect=5)) as resp:
            if resp.status in (200, 204):
                log.info("Discord alert delivered for '%s' @ %s", job.get("title"), job.get("company"))
                return True

            elif resp.status == 429:
                retry_after = (await resp.json()).get("retry_after", 2)
                log.warning("Discord 429 rate limit hit. Sleeping %.1fs", retry_after)
                await asyncio.sleep(retry_after)
                # Retry once
                async with session.post(target_url, json=payload) as retry_resp:
                    return retry_resp.status in (200, 204)
            else:
                resp_text = await resp.text()
                log.warning("Discord webhook returned HTTP %d: %s", resp.status, resp_text[:200])
                return False
    except Exception as exc:
        log.error("Failed to send Discord alert for '%s': %s", job.get("title"), exc)
        return False
    finally:
        if close_session:
            await session.close()


def send_discord_alert(
    job: dict[str, Any],
    evaluation: Optional[JobEvaluation] = None,
    webhook_url: Optional[str] = None,
) -> bool:
    """Synchronous helper wrapper around send_discord_alert_async."""
    if evaluation is None:
        # Fallback dummy evaluation for manual test alerts
        evaluation = JobEvaluation(
            is_match=True,
            visa_sponsorship="Work authorization supported",
        )
    return asyncio.run(send_discord_alert_async(job, evaluation, webhook_url))


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    custom_url = sys.argv[1] if len(sys.argv) > 1 else None
    test_job = {
        "title": "Biomedical Software Engineer (Python & IoT)",
        "company": "GE HealthCare",
        "location": "Bengaluru, Karnataka, India",
        "platform": "linkedin",
        "url": "https://www.linkedin.com/jobs/view/test-biomedical-engineer",
    }
    test_eval = JobEvaluation(
        is_match=True,
        visa_sponsorship="Local Indian role (No sponsorship needed)",
    )
    ok = asyncio.run(send_discord_alert_async(test_job, test_eval, custom_url))
    print(f"Test alert sent: {ok}")
