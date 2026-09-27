"""
Pre-processing, spam filtering, and token optimization layers.

Implements:
  * Layer 1 (Regex Spam Filter): Discards non-target and operational jobs using
    word-boundary regexes (medical coding, sales, field maintenance, calibration, etc.).
  * Layer 2 (Token Optimization): BeautifulSoup HTML/CSS/script stripping and
    truncation to 3,500 characters before sending to Gemini LLM.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Layer 1: Regex Spam Filter
# --------------------------------------------------------------------------

SPAM_KEYWORDS: list[str] = [
    "sales", "bde", "business development", "marketing", "representative",
    "field service", "maintenance", "repair technician", "service engineer",
    "billing", "pharmacist", "receptionist", "clerk", "bpo", "helpdesk",
    "customer support", "voice process", "telecaller",
]

SPAM_PATTERNS: dict[str, list[re.Pattern]] = {
    "Medical/Clerical": [
        re.compile(r"\bmedical\s+coding\b", re.IGNORECASE),
        re.compile(r"\bmedical\s+billing\b", re.IGNORECASE),
        re.compile(r"\bbilling\b", re.IGNORECASE),
        re.compile(r"\bpharmacist\b", re.IGNORECASE),
        re.compile(r"\bhospital\s+operations\b", re.IGNORECASE),
        re.compile(r"\breceptionist\b", re.IGNORECASE),
        re.compile(r"\bclerk\b", re.IGNORECASE),
    ],
    "Customer Support": [
        re.compile(r"\bcustomer\s+(?:service|support|success|care)\b", re.IGNORECASE),
        re.compile(r"\bbpo\b", re.IGNORECASE),
        re.compile(r"\bhelpdesk\b", re.IGNORECASE),
        re.compile(r"\bvoice\s+process\b", re.IGNORECASE),
        re.compile(r"\btelecaller\b", re.IGNORECASE),
    ],
    "Sales/BDE": [
        re.compile(r"\bsales\b", re.IGNORECASE),
        re.compile(r"\bbde\b", re.IGNORECASE),
        re.compile(r"\bbusiness\s+development\b", re.IGNORECASE),
        re.compile(r"\bmarketing\b", re.IGNORECASE),
        re.compile(r"\brepresentative\b", re.IGNORECASE),
        re.compile(r"\bretail\s+(?:manager|associate|sales)\b", re.IGNORECASE),
    ],
    "Field/Maintenance": [
        re.compile(r"\bservice\s+(?:engineer|technician)\b", re.IGNORECASE),
        re.compile(r"\bfield\s+service\b", re.IGNORECASE),
        re.compile(r"\bmaintenance\b", re.IGNORECASE),
        re.compile(r"\brepair\s+technician\b", re.IGNORECASE),
        re.compile(r"\bamc\b", re.IGNORECASE),
    ],
    "Hardware": [
        re.compile(r"\bcalibration\b", re.IGNORECASE),
        re.compile(r"\b(?:lab|biomedical|hardware)\s+technician\b", re.IGNORECASE),
    ],
}


def is_spam_title(title: str) -> tuple[bool, str]:
    """Layer 1: Check if the job title matches any compiled spam pattern.

    Returns:
        (True, reason) if spam, (False, "") if clean.
    """
    if not title:
        return True, "empty title"

    title_clean = title.strip()
    for category, patterns in SPAM_PATTERNS.items():
        for pat in patterns:
            match = pat.search(title_clean)
            if match:
                return True, f"matches {category} spam: '{match.group(0)}'"

    return False, ""


# --------------------------------------------------------------------------
# Layer 2: Token Optimization (BS4 HTML stripping & truncation)
# --------------------------------------------------------------------------

_COLLAPSE_WS = re.compile(r"[ \t]+")
_COLLAPSE_LINES = re.compile(r"\n\s*\n+")
_CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_and_truncate_text(raw_html_or_text: Optional[str], max_chars: int = 3500) -> str:
    """Layer 2: Strip HTML, scripts, CSS, and control characters, then truncate to max_chars.

    Cuts LLM inference time and token usage significantly while preserving
    core technical requirements and protecting against malicious payload injection.
    """
    if not raw_html_or_text:
        return ""

    # Sanitize control characters first
    raw_clean = _CTRL_CHARS.sub("", str(raw_html_or_text))

    # Strip HTML tags, style, and script elements using BeautifulSoup
    soup = BeautifulSoup(raw_clean, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "iframe"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Clean whitespace while preserving line structure
    text = _COLLAPSE_WS.sub(" ", text)
    text = _COLLAPSE_LINES.sub("\n", text)
    text = text.strip()

    if len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


# Standard pipeline alias
clean_html_text = clean_and_truncate_text


# --------------------------------------------------------------------------
# Layer 2B: Semantic Anchoring & Description Hashing
# --------------------------------------------------------------------------

import hashlib

DESC_ANCHORS: list[str] = [
    "qualifications",
    "requirements",
    "what you'll do",
    "what you will do",
    "responsibilities",
    "key responsibilities",
    "role overview",
    "about the role",
    "job description",
    "job summary",
    "basic qualifications",
    "minimum qualifications",
    "preferred qualifications",
    "skills required",
    "what we are looking for",
    "what you bring",
    "position overview",
    "essential duties",
]


def extract_anchored_description(description: str) -> str:
    """Extract description starting from the first semantic anchor to skip boilerplate company intros."""
    if not description or description == "Description not available.":
        return ""

    desc_lower = description.lower()
    best_index = -1

    for anchor in DESC_ANCHORS:
        idx = desc_lower.find(anchor)
        if idx != -1:
            if best_index == -1 or idx < best_index:
                best_index = idx

    if best_index != -1:
        return description[best_index:]
    return description


def generate_desc_hash(description: str) -> str:
    """Creates a deterministic hash of the core job description using semantic anchoring.
    
    Anchors at sections like 'Qualifications', 'Requirements', or 'What you'll do'
    to prevent enterprise company mission boilerplate (e.g. Medtronic/Philips intros)
    from causing false-positive deduplication collisions across different job postings.
    """
    if not description or description == "Description not available.":
        return ""

    anchored = extract_anchored_description(description)
    normalized = re.sub(r"[^a-z0-9]", "", anchored.lower())

    if not normalized:
        normalized = re.sub(r"[^a-z0-9]", "", description.lower())

    if not normalized:
        return ""

    # Hash only the first 500 characters of the anchored content
    return hashlib.sha256(normalized[:500].encode("utf-8")).hexdigest()[:16]



