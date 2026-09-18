"""Shared plumbing every scraper uses.

Keeping normalization here means the four platforms hand Phase 3 exactly
the same shape, whatever their source format looks like.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

import config
from database import is_new_job, make_job_id

log = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z]+")


def clean(text: Optional[str]) -> str:
    """Collapse whitespace, strip, tolerate None."""
    return _WS.sub(" ", (text or "")).strip()


def clean_job_url(url: str, platform: str = "") -> str:
    """Produce clean, canonical job URLs that open reliably without
    broken tracking redirects or auth walls."""
    if not url:
        return ""
    clean_url = url.strip()
    plat = (platform or "").lower()

    # LinkedIn: extract canonical job id to avoid auth-locked tracking URLs
    if "linkedin" in plat or "linkedin.com" in clean_url:
        m = re.search(r"/jobs/view/(?:[^/?#]+-)?(\d+)", clean_url)
        if m:
            return f"https://www.linkedin.com/jobs/view/{m.group(1)}/"
        return clean_url.split("?")[0]

    # Indeed: canonical viewjob link with jk parameter
    if "indeed" in plat or "indeed.com" in clean_url:
        m = re.search(r"[?&]jk=([a-zA-Z0-9]+)", clean_url)
        if m:
            return f"https://www.indeed.com/viewjob?jk={m.group(1)}"
        return clean_url.split("&utm")[0].split("?utm")[0]

    # Naukri: ensure https://www.naukri.com/ with proper slash and clean path
    if "naukri" in plat or "naukri.com" in clean_url:
        if clean_url.startswith(("javascript:", "#")):
            return clean_url
        if not clean_url.startswith("http"):
            if not clean_url.startswith("/"):
                clean_url = "/" + clean_url
            clean_url = f"https://www.naukri.com{clean_url}"
        return clean_url.split("?")[0]

    return clean_url


# --------------------------------------------------------------------------
# Job record
# --------------------------------------------------------------------------

@dataclass
class JobResult:
    title: str
    company: str
    url: str
    platform: str
    location: str = ""
    description: str = ""
    job_id: str = ""
    is_international: bool = False
    posted: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.title = clean(self.title)
        self.company = clean(self.company)
        self.location = clean(self.location)
        self.platform = (self.platform or "").strip().lower()
        self.url = clean_job_url(self.url, self.platform)
        self.is_international = not is_india(self.location)
        if not self.job_id:
            self.job_id = make_job_id(self.platform, self.url, self.title, self.company)

    def is_valid(self) -> bool:
        return bool(self.title and self.company and self.url.startswith("http"))

    def to_row(self) -> dict[str, Any]:
        """Shape expected by database.add_job(). sponsorship_offered and
        tech_matches stay unset — Phase 3 fills those in."""
        row = asdict(self)
        row.pop("raw", None)
        row.pop("posted", None)
        row["date_found"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return row


# --------------------------------------------------------------------------
# Location
# --------------------------------------------------------------------------

def is_india(location: str) -> bool:
    """True when the posting is India-based (so Phase 3 skips the visa check).

    Word-level matching, not substring: 'Indiana' and 'Delhi Township, OH'
    are the traps a naive `'india' in text` check falls into.
    """
    text = (location or "").lower()
    if not text:
        return False
    if "indiana" in text:
        text = text.replace("indiana", " ")
    tokens = set(_WORD.findall(text))
    if tokens & config.INDIA_MARKERS:
        # Guard against 'Delhi, Ohio' style US namesakes.
        if re.search(r"\b(usa|united states|u\.s\.|ohio|ontario|canada)\b", text):
            return False
        return True
    return False


# --------------------------------------------------------------------------
# Filtering helpers
# --------------------------------------------------------------------------

def platform_from_url(url: str, default: str = "") -> str:
    host = urlparse(url or "").netloc.lower()
    for name in config.PLATFORMS:
        if name in host:
            return name
    return default


def dedupe(jobs: Iterable[JobResult]) -> list[JobResult]:
    """Within-run dedup: the same posting shows up under several queries."""
    seen: set[str] = set()
    out: list[JobResult] = []
    for job in jobs:
        if job.job_id in seen:
            continue
        seen.add(job.job_id)
        out.append(job)
    return out


def keep_unseen(jobs: Iterable[JobResult]) -> list[JobResult]:
    """Across-run dedup via the Phase 1 gate. Called before any description
    fetch, so known jobs cost one indexed lookup and nothing else."""
    fresh = [j for j in dedupe(jobs) if is_new_job(j.job_id)]
    log.info("dedup: %d new", len(fresh))
    return fresh


def valid_only(jobs: Iterable[JobResult]) -> list[JobResult]:
    out, dropped = [], 0
    for job in jobs:
        if job.is_valid():
            out.append(job)
        else:
            dropped += 1
    if dropped:
        log.warning("dropped %d malformed job(s) — selectors may have drifted", dropped)
    return out


# --------------------------------------------------------------------------
# Pacing
# --------------------------------------------------------------------------

def sleep_jitter(lo: float, hi: float) -> None:
    time.sleep(random.uniform(lo, hi))


async def asleep_jitter(lo: float, hi: float) -> None:
    await asyncio.sleep(random.uniform(lo, hi))
