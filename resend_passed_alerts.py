import asyncio
import logging
import os
from typing import Optional
from urllib.parse import urlsplit

import aiohttp
import libsql_client
from dotenv import load_dotenv

from discord_alerts import classify_domain_fallback, send_discord_alert_async

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TENANT_MAP = {
    "medtronic": {"domain": "https://medtronic.wd1.myworkdayjobs.com", "tenant": "medtronic", "site": "MedtronicCareers"},
    "philips": {"domain": "https://philips.wd3.myworkdayjobs.com", "tenant": "philips", "site": "jobs-and-careers"},
    "stryker": {"domain": "https://stryker.wd1.myworkdayjobs.com", "tenant": "stryker", "site": "StrykerCareers"},
    "gehc": {"domain": "https://gehc.wd5.myworkdayjobs.com", "tenant": "gehc", "site": "GEHC_ExternalSite"},
    "gehealthcare": {"domain": "https://gehc.wd5.myworkdayjobs.com", "tenant": "gehc", "site": "GEHC_ExternalSite"},
    "getinge": {"domain": "https://getinge.wd3.myworkdayjobs.com", "tenant": "getinge", "site": "Getinge_Careers"},
    "abbott": {"domain": "https://abbott.wd5.myworkdayjobs.com", "tenant": "abbott", "site": "abbottcareers2"},
    "bd": {"domain": "https://bd.wd1.myworkdayjobs.com", "tenant": "bd", "site": "bdcareers"},
    "baxter": {"domain": "https://baxter.wd1.myworkdayjobs.com", "tenant": "baxter", "site": "baxter"},
    "danaher": {"domain": "https://danaher.wd1.myworkdayjobs.com", "tenant": "danaher", "site": "DanaherCareers"},
    "zimmerbiomet": {"domain": "https://zimmerbiomet.wd1.myworkdayjobs.com", "tenant": "zimmerbiomet", "site": "Zimmer_Biomet_Careers"},
    "smithnephew": {"domain": "https://smithnephew.wd5.myworkdayjobs.com", "tenant": "smithnephew", "site": "External"},
}


def resolve_workday_urls(raw_url: str, company: str) -> tuple[str, Optional[str]]:
    """Resolves and repairs Workday candidate URLs to ensure the correct public career site slug.

    Replaces invalid slugs (like '/External/' on Medtronic/Philips/Stryker) with the company's
    actual career portal slug, and returns the CXS endpoint for live availability checks.

    Returns:
        tuple (public_url, cxs_verification_url)
    """
    if not raw_url or "myworkdayjobs.com" not in raw_url:
        return raw_url, None

    if "/job/" not in raw_url:
        return raw_url, None

    job_path = "/job/" + raw_url.split("/job/")[1]
    comp_lower = (company or "").lower()
    url_lower = raw_url.lower()

    for key, cfg in TENANT_MAP.items():
        if key in url_lower or key in comp_lower:
            public_url = f"{cfg['domain']}/en-US/{cfg['site']}{job_path}"
            cxs_url = f"{cfg['domain']}/wday/cxs/{cfg['tenant']}/{cfg['site']}{job_path}"
            return public_url, cxs_url

    # Generic fallback
    if "/wday/cxs/" in raw_url:
        parts = raw_url.split("/wday/cxs/")
        base_domain = parts[0]
        sub_parts = parts[1].split("/")
        if len(sub_parts) >= 3:
            site_name = sub_parts[1]
            return f"{base_domain}/en-US/{site_name}{job_path}", None

    return raw_url, None


async def resend_all_passed_jobs():
    db_url = os.getenv("TURSO_DATABASE_URL")
    db_token = os.getenv("TURSO_AUTH_TOKEN")

    if not db_url or not db_token:
        logger.error("Turso credentials missing from .env!")
        return

    # LibSQL client requires https:// or wss://; replace libsql:// with https://
    client_url = db_url.replace("libsql://", "https://") if db_url.startswith("libsql://") else db_url
    client = libsql_client.create_client(url=client_url, auth_token=db_token)

    try:
        result = await client.execute(
            """
            SELECT job_id AS id, title, company, location, platform, url, description, 
                   ai_score, visa_sponsorship, match_reason, job_category, tier
            FROM job_postings 
            WHERE status IN ('active', 'consensus_passed')
            ORDER BY ai_score DESC
            """
        )

        passed_jobs = result.rows
        logger.info(f"Loaded {len(passed_jobs)} previously approved jobs from Turso.")

        sent_count = 0
        skipped_stale = 0
        fixed_urls = 0
        channel_counts: dict[str, int] = {}

        async with aiohttp.ClientSession() as http_session:
            for idx, row in enumerate(passed_jobs, 1):
                raw_url = row[5] if row[5] else ""
                company_name = row[2] or ""
                repaired_url, cxs_url = resolve_workday_urls(raw_url, company_name)

                # Pre-flight Live Verification for Workday jobs to prevent 404 / 'Page Not Found'
                if cxs_url:
                    try:
                        async with http_session.get(
                            cxs_url,
                            headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
                            timeout=aiohttp.ClientTimeout(total=4),
                        ) as cxs_resp:
                            if cxs_resp.status in (403, 404):
                                skipped_stale += 1
                                logger.warning(
                                    f"[{idx}/{len(passed_jobs)}] Skipping CLOSED/EXPIRED posting (employer removed role): "
                                    f"'{row[1]}' @ {company_name} (HTTP {cxs_resp.status})"
                                )
                                # Mark as expired in DB so it won't be queried again
                                try:
                                    await client.execute(
                                        "UPDATE job_postings SET status = 'expired' WHERE job_id = ?",
                                        [row[0]],
                                    )
                                except Exception:
                                    pass
                                continue
                    except Exception as verify_err:
                        logger.debug(f"Verification timeout on {cxs_url}: {verify_err}")

                if repaired_url != raw_url:
                    fixed_urls += 1
                    logger.info(f"Healed Workday URL for '{row[1]}' @ {company_name}: {raw_url} -> {repaired_url}")
                    try:
                        await client.execute(
                            "UPDATE job_postings SET url = ? WHERE job_id = ?",
                            [repaired_url, row[0]],
                        )
                    except Exception as update_err:
                        logger.debug(f"Could not update database URL: {update_err}")

                category = row[10] if row[10] else "General"
                if category == "General":
                    inferred = classify_domain_fallback(row[1], company_name)
                    if inferred != "General":
                        category = inferred
                        try:
                            await client.execute(
                                "UPDATE job_postings SET job_category = ? WHERE job_id = ?",
                                [category, row[0]],
                            )
                        except Exception as cat_err:
                            logger.debug(f"Could not update database job_category: {cat_err}")

                channel_target = {
                    "Biomedical_RD": "#biomed-chat",
                    "ECE_Hardware": "#ece-chat",
                    "Software_Web": "#software-chat",
                    "General": "#general-chat",
                }.get(category, "#general-chat")

                job_data = {
                    "id": row[0],
                    "job_id": row[0],
                    "title": row[1],
                    "company": company_name,
                    "location": row[3],
                    "platform": row[4],
                    "url": repaired_url,
                    "description": row[6] if row[6] is not None else "",
                    "ai_score": row[7] if row[7] is not None else 75,
                    "visa_sponsorship": row[8] if row[8] else "Not Specified",
                    "match_reason": row[9] if row[9] else "Previously evaluated match.",
                    "job_category": category,
                    "tier": row[11] if row[11] else "strict",
                    "is_match": True,
                }

                logger.info(
                    f"[{idx}/{len(passed_jobs)}] Resending [LIVE VERIFIED]: {job_data['title']} @ {job_data['company']} "
                    f"(Score: {job_data['ai_score']}) -> {category} ({channel_target})"
                )

                success = await send_discord_alert_async(job_data)
                if success:
                    sent_count += 1
                    channel_counts[channel_target] = channel_counts.get(channel_target, 0) + 1
                    try:
                        await client.execute(
                            "UPDATE job_postings SET alert_sent = 1 WHERE job_id = ?",
                            [row[0]],
                        )
                    except Exception:
                        pass

                # Respect Discord rate limits
                await asyncio.sleep(1.2)

        logger.info(
            f"FINISHED: Resent {sent_count} live-verified alerts across channels: {channel_counts}. "
            f"Healed {fixed_urls} Workday URLs. Pruned {skipped_stale} closed/expired postings."
        )

    except Exception as e:
        logger.error(f"Error during resend: {e}", exc_info=True)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(resend_all_passed_jobs())
