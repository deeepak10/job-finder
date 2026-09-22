"""Central configuration for the pipeline.

Everything tunable lives here so the scrapers stay logic-only and you can
retune searches without touching code paths that are already tested.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:  # dotenv optional in CI, where GitHub Secrets are env vars
    pass

# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------
SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "").strip()
APIFY_TOKEN: str = os.getenv("APIFY_TOKEN", "").strip()
RAPIDAPI_KEY: str = os.getenv("RAPIDAPI_KEY", "").strip()
WELLFOUND_ACTOR_ID: str = os.getenv("WELLFOUND_ACTOR_ID", "").strip()
LINKEDIN_ACTOR_ID: str = os.getenv("LINKEDIN_ACTOR_ID", "").strip()
NAUKRI_ACTOR_ID: str = os.getenv("NAUKRI_ACTOR_ID", "").strip()
ADZUNA_APP_ID: str = os.getenv("ADZUNA_APP_ID", "").strip()
ADZUNA_APP_KEY: str = os.getenv("ADZUNA_APP_KEY", "").strip()
DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
# The original webhook is now the fallback/default
DISCORD_WEBHOOK_DEFAULT: str = DISCORD_WEBHOOK_URL

# New Category Webhooks (Multi-Channel Control Room)
DISCORD_WEBHOOK_BIOMED: str = os.getenv("DISCORD_WEBHOOK_BIOMED", "").strip()
DISCORD_WEBHOOK_ECE: str = os.getenv("DISCORD_WEBHOOK_ECE", "").strip()
DISCORD_WEBHOOK_SOFTWARE: str = os.getenv("DISCORD_WEBHOOK_SOFTWARE", "").strip()

# Turso Cloud Database
TURSO_DATABASE_URL: str = os.getenv("TURSO_DATABASE_URL", "").strip()
TURSO_AUTH_TOKEN: str = os.getenv("TURSO_AUTH_TOKEN", "").strip()

# Gemini LLM Evaluation — Production gemini-3.6-flash endpoint
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_TIMEOUT_SECONDS: float = 45.0

# Groq Fallback LLM Evaluation
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
GROQ_ENSEMBLE_MODEL: str = os.getenv("GROQ_ENSEMBLE_MODEL", "openai/gpt-oss-120b").strip()

# OpenRouter Ensemble Evaluation (Zero-Cost Free Tier)
OR_MODEL: str = os.getenv("OR_MODEL", "meta-llama/llama-3.2-3b-instruct:free").strip()
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", OR_MODEL).strip()

# Priority Healthcare Companies (triggers Purple alert color: 986895)
TARGET_HEALTHCARE_COMPANIES: list[str] = [
    "Skanray Technologies",
    "Stryker",
    "Dozee",
    "GE HealthCare",
    "Philips",
    "Qure.ai",
    "Agappe Diagnostics",
    "Schiller Healthcare",
    "Schiller",
    "BPL Medical Technologies",
    "BPL Medical",
    "Getinge",
]


def mask_secret(value: str, prefix_len: int = 4, suffix_len: int = 4) -> str:
    """Mask sensitive credentials for safe logging and status reporting."""
    if not value:
        return "<NOT SET>"
    if len(value) <= (prefix_len + suffix_len):
        return "***"
    return f"{value[:prefix_len]}...{value[-suffix_len:]}"


def validate_configuration() -> dict[str, Any]:
    """Validate presence and format of critical runtime settings without leaking secrets."""
    status = {
        "turso_configured": bool(TURSO_DATABASE_URL and TURSO_AUTH_TOKEN),
        "gemini_configured": bool(GEMINI_API_KEY),
        "discord_configured": bool(DISCORD_WEBHOOK_URL),
        "discord_biomed_configured": bool(DISCORD_WEBHOOK_BIOMED),
        "discord_ece_configured": bool(DISCORD_WEBHOOK_ECE),
        "discord_software_configured": bool(DISCORD_WEBHOOK_SOFTWARE),
        "serpapi_configured": bool(SERPAPI_API_KEY),
        "apify_configured": bool(APIFY_TOKEN),
        "groq_configured": bool(GROQ_API_KEY),
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "issues": [],
    }

    if not TURSO_DATABASE_URL:
        status["issues"].append("TURSO_DATABASE_URL is missing.")
    elif not (TURSO_DATABASE_URL.startswith("libsql://") or TURSO_DATABASE_URL.startswith("https://")):
        status["issues"].append("TURSO_DATABASE_URL must begin with 'libsql://' or 'https://'.")

    if not TURSO_AUTH_TOKEN:
        status["issues"].append("TURSO_AUTH_TOKEN is missing.")

    if not GEMINI_API_KEY:
        status["issues"].append("GEMINI_API_KEY is missing.")

    for name, hook in [
        ("DISCORD_WEBHOOK_URL", DISCORD_WEBHOOK_URL),
        ("DISCORD_WEBHOOK_BIOMED", DISCORD_WEBHOOK_BIOMED),
        ("DISCORD_WEBHOOK_ECE", DISCORD_WEBHOOK_ECE),
        ("DISCORD_WEBHOOK_SOFTWARE", DISCORD_WEBHOOK_SOFTWARE),
    ]:
        if hook and not hook.startswith(
            ("https://discord.com/api/webhooks/", "https://discordapp.com/api/webhooks/")
        ):
            status["issues"].append(f"{name} must start with 'https://discord.com/api/webhooks/'.")

    return status



# --------------------------------------------------------------------------
# Target Scraper Queries (Strict MedTech vs Broad Generic)
# --------------------------------------------------------------------------
TARGET_QUERIES = [
    "Medical Device R&D",
    "Biomedical Firmware",
    "Medical IoT",
    "IoT Medical Devices",
    "Python Healthtech",
    "Signal Processing Engineer",
    "R&D Engineer Medical",
    "Schiller ECG firmware",
    "Schiller Healthcare R&D",
    "BPL Medical R&D",
    "BPL Medical firmware",
]

STRICT_QUERIES = [
    "Medical Device R&D",
    "Biomedical Firmware",
    "Medical IoT",
    "IoT Medical Devices",
    "Signal Processing Engineer",
    "R&D Engineer Medical",
    "Schiller ECG firmware",
    "Schiller Healthcare R&D",
    "BPL Medical R&D",
    "BPL Medical firmware",
]

BROAD_QUERIES = [
    "Embedded Software Engineer Healthcare",
    "Python Developer Healthcare",
    "Research and Development Engineer",
    "Python Healthtech",
]

# --------------------------------------------------------------------------
# Hybrid Aggregator Keywords & URLs (Tier 1: Sniper vs Tier 2: Constrained Net)
# --------------------------------------------------------------------------
TARGET_URLS = [
    # Tier 1: The Sniper (Rigid / Hyper-Specific)
    "https://www.naukri.com/medical-device-rnd-jobs",
    "https://www.naukri.com/biomedical-firmware-jobs",
    "https://www.naukri.com/medical-iot-jobs",
    "https://www.naukri.com/signal-processing-engineer-jobs",
    "https://www.naukri.com/schiller-ecg-firmware-jobs",
    "https://www.naukri.com/schiller-healthcare-rnd-jobs",
    "https://www.naukri.com/bpl-medical-rnd-jobs",
    "https://www.naukri.com/bpl-medical-firmware-jobs",

    # Tier 2: The Constrained Net (Generic Title + Domain Modifier)
    "https://www.naukri.com/embedded-software-engineer-healthcare-jobs",
    "https://www.naukri.com/software-engineer-medical-device-jobs",
    "https://www.naukri.com/hardware-software-interfacing-jobs",
    "https://www.naukri.com/python-developer-healthcare-jobs",
]

NAUKRI_STRICT_SEARCHES = [
    ("medical-device-rnd-jobs", "Medical Device R&D"),
    ("biomedical-firmware-jobs", "Biomedical Firmware"),
    ("medical-iot-jobs", "Medical IoT"),
    ("iot-medical-devices-jobs", "IoT Medical Devices"),
    ("signal-processing-engineer-jobs", "Signal Processing Engineer"),
    ("rnd-engineer-medical-jobs", "R&D Engineer Medical"),
    ("schiller-ecg-firmware-jobs", "Schiller ECG firmware"),
    ("schiller-healthcare-rnd-jobs", "Schiller Healthcare R&D"),
    ("bpl-medical-rnd-jobs", "BPL Medical R&D"),
    ("bpl-medical-firmware-jobs", "BPL Medical firmware"),
]

NAUKRI_BROAD_SEARCHES = [
    ("embedded-software-engineer-healthcare-jobs", "Embedded Software Engineer Healthcare"),
    ("software-engineer-medical-device-jobs", "Software Engineer Medical Device"),
    ("hardware-software-interfacing-jobs", "Hardware Software Interfacing"),
    ("python-developer-healthcare-jobs", "Python Developer Healthcare"),
    ("python-healthtech-jobs", "Python Healthtech"),
    ("research-and-development-engineer-jobs", "Research and Development Engineer"),
]

NAUKRI_SEARCHES = NAUKRI_STRICT_SEARCHES + NAUKRI_BROAD_SEARCHES


def get_query_tier(slug_or_query: str) -> str:
    """Determine whether a search slug or query string belongs to 'strict' or 'broad' tier."""
    s = (slug_or_query or "").lower().strip()
    broad_indicators = [
        "embedded-software-engineer-healthcare",
        "software-engineer-medical-device",
        "hardware-software-interfacing",
        "python-developer-healthcare",
        "research-and-development-engineer",
        "python-healthtech",
        "embedded software engineer healthcare",
        "software engineer medical device",
        "hardware software interfacing",
        "python developer healthcare",
        "research and development engineer",
        "python healthtech",
    ]
    for b in broad_indicators:
        if b in s:
            return "broad"
    return "strict"


NAUKRI_BASE = "https://www.naukri.com"
NAUKRI_MAX_PAGES = 1  # Strictly limited to 1 page for freshest listings
NAUKRI_MIN_DELAY = 2.0   # seconds between page loads
NAUKRI_MAX_DELAY = 4.0
NAUKRI_TIMEOUT_MS = 30_000  # Strict 30-second timeout on page loads

# --------------------------------------------------------------------------
# Scraper Toggles & Global Configuration
# --------------------------------------------------------------------------
NAUKRI_ENABLED = True
GOOGLE_JOBS_ENABLED = True
WELLFOUND_ENABLED = True
LINKEDIN_ENABLED = True
ADZUNA_ENABLED = True
WORKDAY_ENABLED = True

# --------------------------------------------------------------------------
# Google Jobs (SerpApi) — covers LinkedIn + Indeed
# --------------------------------------------------------------------------
GOOGLE_JOBS_QUERIES = TARGET_QUERIES
GOOGLE_JOBS_QUERY = TARGET_QUERIES[0]
SERPAPI_MAX_PAGES = 1  # Hardcode pagination limit to max 1 page (max 20 results)
SERPAPI_MAX_RESULTS_PER_QUERY = 20
# Regional & Tier-2 Engineering Hubs (Prevents missing local hardware startups)
REGIONAL_HUBS = [
    "Kerala, India",
    "Kochi, Kerala, India",
    "Kozhikode, Kerala, India",
    "Thiruvananthapuram, Kerala, India",
    "Coimbatore, Tamil Nadu, India"
]

# Major National Tech Hubs
NATIONAL_HUBS = [
    "Bengaluru, Karnataka, India",
    "Pune, Maharashtra, India",
    "Hyderabad, Telangana, India",
    "Chennai, Tamil Nadu, India"
]

# Strategic Global MedTech Epicenters
GLOBAL_HUBS = [
    "Galway, Ireland",          # Medtronic, Boston Scientific R&D
    "Eindhoven, Netherlands",   # Philips Healthcare Global HQ
    "Minneapolis, MN",          # Medical Alley / Device capital
    "Boston, MA",               # HealthTech & Biomedical hardware
    "Munich, Germany"           # Siemens Healthineers & Embedded engineering
]

# Master Search Pool
SERPAPI_LOCATIONS = REGIONAL_HUBS + NATIONAL_HUBS + GLOBAL_HUBS

INDIA_LOCATIONS = [
    "Kochi, Kerala, India",
    "Kozhikode, Kerala, India",
    "Thiruvananthapuram, Kerala, India",
    "Kanhangad, Kerala, India",
    "Kasaragod, Kerala, India",
    "Kannur, Kerala, India",
    "Malappuram, Kerala, India",
    "Palakkad, Kerala, India",
    "Thrissur, Kerala, India",
    "Kollam, Kerala, India",
    "Alappuzha, Kerala, India",
    "Kottayam, Kerala, India",
    "Idukki, Kerala, India",
    "Wayanad, Kerala, India",
    "Pathanamthitta, Kerala, India",
    "Delhi, India",
    "Noida, Uttar Pradesh, India",
    "Bengaluru, Karnataka, India",
    "Hyderabad, Telangana, India",
    "Pune, Maharashtra, India",
    "Chennai, Tamil Nadu, India",
    "Kolkata, West Bengal, India",
]
GLOBAL_LOCATIONS = [
    "Germany",
    "Netherlands",
    "Ireland",
    "United Kingdom",
    "Singapore",
    "Dubai,UAE",
    "Doha,Qatar",
    "Muscat,Oman",
    "Riyadh,Saudi Arabia",
    "Canada",
    "San Francisco, CA",
    "New York, NY",
    "USA",
]
# Only these two boards are wanted out of Google Jobs' aggregate results.
ALLOWED_JOB_BOARDS = ("linkedin", "indeed")

# --------------------------------------------------------------------------
# Wellfound
# --------------------------------------------------------------------------
WELLFOUND_INDUSTRY_TAGS = ["Health Tech", "Medical Devices"]
WELLFOUND_KEYWORDS = TARGET_QUERIES
WELLFOUND_SKILLS = ["Python", "Hardware", "Data"]
WELLFOUND_MAX_RESULTS = 20  # Cap at top 20 latest results per run

# --------------------------------------------------------------------------
# Location classification (feeds the Phase 3 visa filter)
# --------------------------------------------------------------------------
INDIA_MARKERS = {
    "india", "bharat", "bengaluru", "bangalore", "hyderabad", "pune",
    "chennai", "mumbai", "delhi", "noida", "gurugram", "gurgaon",
    "kochi", "cochin", "kerala", "kozhikode", "thiruvananthapuram",
    "trivandrum", "kanhangad", "mangalore", "mangaluru", "coimbatore",
    "ahmedabad", "kolkata", "jaipur", "indore", "chandigarh", "vizag",
    "visakhapatnam", "nagpur", "surat", "lucknow", "bhubaneswar",
}

PLATFORMS = ("naukri", "wellfound", "linkedin", "indeed", "adzuna", "workday")
