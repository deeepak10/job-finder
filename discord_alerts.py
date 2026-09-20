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
    evaluation: Any,
) -> dict[str, Any]:
    """Construct the Discord Rich Embed."""
    title = _safe_str(job.get("title"), max_len=240, fallback="Job Posting")
    url = (job.get("url") or "").strip()
    company = _safe_str(job.get("company"), max_len=200, fallback="—")
    location = _safe_str(job.get("location"), max_len=200, fallback="—")
    platform = _safe_str((job.get("platform") or "Web").capitalize(), max_len=50, fallback="Web")
    color = pick_embed_color(company)

    match_reason = (
        (evaluation.get("match_reason") if isinstance(evaluation, dict) else getattr(evaluation, "match_reason", ""))
        or job.get("match_reason")
        or "Passed semantic evaluation."
    )

    visa_sponsorship = (
        evaluation.get("visa_sponsorship")
        if isinstance(evaluation, dict)
        else getattr(evaluation, "visa_sponsorship", "Not Specified")
    )

    raw_category = None
    if evaluation and isinstance(evaluation, dict):
        raw_category = evaluation.get("job_category")
    elif hasattr(evaluation, "job_category"):
        raw_category = getattr(evaluation, "job_category", None)
    elif job and isinstance(job, dict):
        raw_category = job.get("job_category")

    category = raw_category.strip() if isinstance(raw_category, str) and raw_category.strip() else "General"

    fields = [
        {"name": "🏢 Company", "value": company, "inline": True},
        {"name": "📍 Location", "value": location, "inline": True},
        {"name": "🌐 Platform", "value": platform, "inline": True},
        {"name": "🏷️ Domain", "value": _safe_str(category.replace("_", " "), max_len=50, fallback="General"), "inline": True},
        {
            "name": "🛂 Visa Status",
            "value": _safe_str(visa_sponsorship, max_len=200, fallback="Not Specified"),
            "inline": True,
        },
        {
            "name": "Why it matched:",
            "value": _safe_str(match_reason, max_len=1000, fallback="Passed semantic evaluation."),
            "inline": False,
        },
    ]

    embed = {
        "title": f"🟩 {title}",
        "url": url,
        "color": color,
        "fields": fields,
        "footer": {
            "text": "Tri-Model AI Routing Pipeline",
        },
    }

    return embed


async def send_discord_alert_async(
    job: dict[str, Any],
    evaluation: Any,
    webhook_url: Optional[str] = None,
    session: Optional[aiohttp.ClientSession] = None,
) -> bool:
    """Send alert to Discord webhook asynchronously with multi-channel routing.

    Extracts job_category from AI evaluation and dynamically maps it to
    channel-specific webhooks (Biomedical, ECE, Software, or General).
    Enforces gating: Only dispatches if is_match is True.
    """
    is_match = False
    if isinstance(evaluation, dict):
        is_match = bool(evaluation.get("is_match", False))
    elif hasattr(evaluation, "is_match"):
        is_match = bool(getattr(evaluation, "is_match", False))

    if not is_match:
        log.info(
            "Skipping Discord alert for '%s' (is_match=%s)",
            job.get("title"),
            is_match,
        )
        return False

    # Extract category from the AI evaluation or fallback to General
    raw_category = None
    if evaluation and isinstance(evaluation, dict):
        raw_category = evaluation.get("job_category")
    elif hasattr(evaluation, "job_category"):
        raw_category = getattr(evaluation, "job_category", None)
    elif job and isinstance(job, dict):
        raw_category = job.get("job_category")

    category = raw_category.strip() if isinstance(raw_category, str) and raw_category.strip() else "General"

    # Map the AI's category to the specific webhook
    webhook_map = {
        "Biomedical_RD": getattr(config, "DISCORD_WEBHOOK_BIOMED", ""),
        "ECE_Hardware": getattr(config, "DISCORD_WEBHOOK_ECE", ""),
        "Software_Web": getattr(config, "DISCORD_WEBHOOK_SOFTWARE", ""),
        "General": getattr(config, "DISCORD_WEBHOOK_DEFAULT", "") or getattr(config, "DISCORD_WEBHOOK_URL", ""),
    }

    default_webhook = (
        getattr(config, "DISCORD_WEBHOOK_DEFAULT", "")
        or getattr(config, "DISCORD_WEBHOOK_URL", "")
        or DISCORD_WEBHOOK_URL
    )

    # Specific argument takes top priority; otherwise route by category with default fallback
    target_webhook = webhook_map.get(category, default_webhook)
    if not target_webhook:
        target_webhook = default_webhook

    target_url = webhook_url or target_webhook
    if not target_url:
        log.warning("No Discord Webhook configured. Cannot send alert.")
        return False

    # SSRF protection: only allow official Discord webhook endpoints
    if not target_url.startswith(
        ("https://discord.com/api/webhooks/", "https://discordapp.com/api/webhooks/")
    ):
        log.error("Security alert: Target Discord webhook must point to official discord.com endpoints.")
        return False

    embed = build_discord_embed(job, evaluation)
    payload = {"embeds": [embed]}

    # Anti-burst rate limit protection: 1.5 second sleep guarantees staying well below Discord limits
    await asyncio.sleep(1.5)

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True

    try:
        async with session.post(target_url, json=payload, timeout=aiohttp.ClientTimeout(total=15, connect=5)) as resp:
            if resp.status in (200, 204):
                log.info("Successfully routed alert to %s channel for '%s' @ %s", category, job.get("title"), job.get("company"))
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
                log.warning("Failed to route alert to %s channel. Status: %d: %s", category, resp.status, resp_text[:200])
                return False
    except Exception as exc:
        log.error("Discord routing error for '%s' (%s): %s", job.get("title"), category, exc)
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


async def send_system_alert_async(
    message: str,
    webhook_url: Optional[str] = None,
    session: Optional[aiohttp.ClientSession] = None,
) -> bool:
    """Fires a red-colored embed to Discord if the pipeline experiences a critical anomaly."""
    target_url = webhook_url or config.DISCORD_WEBHOOK_URL or DISCORD_WEBHOOK_URL
    if not target_url:
        log.warning("DISCORD_WEBHOOK_URL not set; skipping system notification.")
        return False

    if not target_url.startswith(
        ("https://discord.com/api/webhooks/", "https://discordapp.com/api/webhooks/")
    ):
        log.error("Security alert: DISCORD_WEBHOOK_URL must point to official discord.com endpoints.")
        return False

    embed = {
        "title": "⚠️ Pipeline Anomaly Detected",
        "description": message,
        "color": 16711680,  # Red
        "footer": {"text": "Job Finder System Monitor"},
    }
    payload = {"embeds": [embed]}

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True

    try:
        async with session.post(target_url, json=payload, timeout=aiohttp.ClientTimeout(total=15, connect=5)) as resp:
            if resp.status in (200, 204):
                log.info("System alert delivered to Discord: %s", message[:100])
                return True
            else:
                resp_text = await resp.text()
                log.warning("System alert webhook returned HTTP %d: %s", resp.status, resp_text[:200])
                return False
    except Exception as exc:
        log.error("Failed to send system alert to Discord: %s", exc)
        return False
    finally:
        if close_session:
            await session.close()


def send_system_alert(
    message: str,
    webhook_url: Optional[str] = None,
) -> bool:
    """Synchronous helper wrapper around send_system_alert_async."""
    return asyncio.run(send_system_alert_async(message, webhook_url))


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
