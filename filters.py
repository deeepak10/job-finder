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

SPAM_PATTERNS: dict[str, list[re.Pattern]] = {
    "Medical/Clerical": [
        re.compile(r"\bmedical\s+coding\b", re.IGNORECASE),
        re.compile(r"\bmedical\s+billing\b", re.IGNORECASE),
        re.compile(r"\bpharmacist\b", re.IGNORECASE),
        re.compile(r"\bhospital\s+operations\b", re.IGNORECASE),
    ],
    "Sales/BDE": [
        re.compile(r"\bsales\s+(?:executive|manager|representative)\b", re.IGNORECASE),
        re.compile(r"\bbde\b", re.IGNORECASE),
        re.compile(r"\bbusiness\s+development\b", re.IGNORECASE),
        re.compile(r"\bretail\s+(?:manager|associate|sales)\b", re.IGNORECASE),
    ],
    "Customer Support": [
        re.compile(r"\bcustomer\s+(?:service|support|success|care)\b", re.IGNORECASE),
    ],
    "Field/Maintenance": [
        re.compile(r"\bservice\s+(?:engineer|technician)\b", re.IGNORECASE),
        re.compile(r"\bfield\s+service\b", re.IGNORECASE),
        re.compile(r"\bmaintenance\s+(?:engineer|technician)\b", re.IGNORECASE),
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


