"""Central configuration for the pipeline.

Everything tunable lives here so the scrapers stay logic-only and you can
retune searches without touching code paths that are already tested.
"""

from __future__ import annotations

import os
from pathlib import Path

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
ADZUNA_APP_ID: str = os.getenv("ADZUNA_APP_ID", "").strip()
ADZUNA_APP_KEY: str = os.getenv("ADZUNA_APP_KEY", "").strip()
DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

# Turso Cloud Database
TURSO_DATABASE_URL: str = os.getenv("TURSO_DATABASE_URL", "").strip()
TURSO_AUTH_TOKEN: str = os.getenv("TURSO_AUTH_TOKEN", "").strip()

# Gemini LLM Evaluation — Strictly targets gemini-3.6-flash to prevent 404/503 errors
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL: str = "gemini-3.6-flash"
GEMINI_TIMEOUT_SECONDS: float = 45.0

# Groq Fallback LLM Evaluation (Llama-3.3-70B)
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()

# Priority Healthcare Companies (triggers Purple alert color: 986895)
TARGET_HEALTHCARE_COMPANIES: list[str] = [
    "Skanray Technologies",
    "Stryker",
    "Dozee",
    "GE HealthCare",
    "Philips",
    "Qure.ai",
    "Agappe Diagnostics",
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
        "serpapi_configured": bool(SERPAPI_API_KEY),
        "apify_configured": bool(APIFY_TOKEN),
        "groq_configured": bool(GROQ_API_KEY),
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

    if DISCORD_WEBHOOK_URL and not DISCORD_WEBHOOK_URL.startswith(
        ("https://discord.com/api/webhooks/", "https://discordapp.com/api/webhooks/")
    ):
        status["issues"].append("DISCORD_WEBHOOK_URL must start with 'https://discord.com/api/webhooks/'.")

    return status



# --------------------------------------------------------------------------
# Target Scraper Queries (Functional R&D Skills)
# --------------------------------------------------------------------------
TARGET_QUERIES = [
    "Medical Device R&D",
    "Biomedical Firmware",
    "Medical IoT",
    "IoT Medical Devices",
    "Python Healthtech",
    "Signal Processing Engineer",
    "R&D Engineer Medical",
]

# --------------------------------------------------------------------------
# Naukri
# --------------------------------------------------------------------------
NAUKRI_SEARCHES = [
    ("medical-device-rnd-jobs", "Medical Device R&D"),
    ("biomedical-firmware-jobs", "Biomedical Firmware"),
    ("medical-iot-jobs", "Medical IoT"),
    ("iot-medical-devices-jobs", "IoT Medical Devices"),
    ("python-healthtech-jobs", "Python Healthtech"),
    ("signal-processing-engineer-jobs", "Signal Processing Engineer"),
    ("rnd-engineer-medical-jobs", "R&D Engineer Medical"),
]
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
