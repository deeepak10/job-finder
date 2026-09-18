# 🩺 Autonomous MedTech & Firmware Engineering Intelligence Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Database: Turso](https://img.shields.io/badge/database-Turso%20(LibSQL)-00EB8D.svg?logo=sqlite&logoColor=black)](https://turso.tech/)
[![Primary LLM: Gemini](https://img.shields.io/badge/primary%20LLM-Gemini%203.6%20Flash-4285F4.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![Fallback LLM: Groq](https://img.shields.io/badge/fallback%20LLM-Groq%20Llama--3.1--8B-F55036.svg?logo=meta&logoColor=white)](https://groq.com/)
[![Browser Automation](https://img.shields.io/badge/automation-Playwright%20Chromium-2EAD33.svg?logo=playwright&logoColor=white)](https://playwright.dev/)
[![CI/CD: GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions%20(Hardened)-2088FF.svg?logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![Tests: Pytest](https://img.shields.io/badge/tests-112%20passed-brightgreen.svg?logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An enterprise-grade, asynchronous Python intelligence pipeline designed to autonomously scrape, deduplicate, filter, semantically evaluate, and deliver real-time alerts for **Biomedical R&D**, **Medical Device Firmware**, **HealthTech**, and **Embedded Systems** engineering opportunities across global epicenters and regional Indian engineering hubs.

Engineered with a **resilient multi-provider LLM waterfall** (Google Gemini &rarr; Groq Llama-3.1-8B), **zero-cost direct enterprise ATS ingestion** (Medtronic, Philips), **quota time-gating**, **serverless cloud persistence** (Turso Cloud LibSQL), and **async Discord notification dispatch**.

---

## 🏗 System Architecture

```mermaid
flowchart TD
    subgraph S["1. Concurrent Scraper Engine"]
        WKD["Workday ATS (Medtronic & Philips Direct)"]
        NAU["Naukri India (Playwright Headless)"]
        ADZ["Adzuna API (Free REST Search)"]
        SERP["SerpApi Google Jobs (Clustered Rotation)"]
        LNK["LinkedIn Jobs (Apify Actor)"]
        WEL["Wellfound Startups (Apify Actor)"]
    end

    subgraph TG["2. Execution Time-Gating & Schedule Alignment"]
        GATE{"UTC Run Hour <= 4?<br/>(07:30 AM IST)"}
        FREE_FLOW["Run On Every Scheduled Run"]
        QUOTA_FLOW["Run Only During Morning Peak"]
    end

    subgraph L0L2["3. Pre-Processing, Dedup & Token Optimization"]
        TURSO_DEDUP{"Layer 0: Seen in Turso?<br/>(Indexed URL or Title+Company Hash)"}
        REGEX_SPAM{"Layer 1: Spam Title Regex?<br/>(Field service, technician, sales)"}
        TOKEN_OPT["Layer 2: BS4 Cleaning & 3,500-Char Capping"]
        DISCARD["Bypassed / Discarded"]
    end

    subgraph EVAL["4. Dual-Provider LLM Waterfall (Paced Evaluation)"]
        GEMINI["Primary: Gemini 3.6 Flash<br/>(20 daily requests, 4.5s pacing)"]
        FALLBACK_CHECK{"429 Quota Exceeded?"}
        GROQ["Fallback: Groq Llama-3.1-8B<br/>(12K TPM cap, strict 6s pacing)"]
    end

    subgraph PERSIST_ALERT["5. Persistence & Delivery"]
        TURSO[(Turso Cloud SQLite DB)]
        MATCH_GATE{"is_match == True?"}
        DISCORD["Discord Webhook Engine<br/>(🟪 Purple: Target HealthTech | 🟦 Blue: Standard)"]
        STORE_ONLY["Stored Without Alert"]
    end

    %% Scraper time-gating connections
    NAU & ADZ & WKD --> FREE_FLOW --> TURSO_DEDUP
    SERP & LNK & WEL --> GATE
    GATE -->|Yes (Morning)| QUOTA_FLOW --> TURSO_DEDUP
    GATE -->|No (Off-Peak)| DISCARD

    %% Pipeline flow
    TURSO_DEDUP -->|Already Seen| DISCARD
    TURSO_DEDUP -->|Fresh Job| REGEX_SPAM
    REGEX_SPAM -->|Spam Match| DISCARD
    REGEX_SPAM -->|Clean Candidate| TOKEN_OPT
    TOKEN_OPT --> GEMINI
    GEMINI --> FALLBACK_CHECK
    FALLBACK_CHECK -->|Success| TURSO
    FALLBACK_CHECK -->|HTTP 429| GROQ
    GROQ --> TURSO
    TURSO --> MATCH_GATE
    MATCH_GATE -->|Yes| DISCORD
    MATCH_GATE -->|No| STORE_ONLY
```

---

## ⚡ Key Engineering Highlights

### 1. Zero-Cost Direct Workday ATS Ingestion
* Queries unauthenticated REST career endpoints for Tier-1 MedTech conglomerates (**Medtronic**, **Philips**) before postings hit third-party aggregators.
* Executes a primary search `POST` followed by an automated secondary detail `GET` request (`jobPostingInfo.jobDescription`) to hydrate the complete posting text for LLM semantic evaluation.

### 2. Multi-Provider LLM Waterfall (Gemini &rarr; Groq Llama-3.1-8B)
* **Primary Stage:** Evaluates candidates through Google Gemini (`gemini-3.6-flash`) with structured Pydantic schemas and 4.5-second pacing delay to exhaust the 20 free daily requests.
* **Autonomous Failover:** Catches `429 RESOURCE_EXHAUSTED` and immediately switches the evaluation queue to Groq Cloud (`llama-3.1-8b-instant`).
* **Rate-Limit Pacer:** Enforces a strict 6-second `asyncio.sleep()` per evaluation on Groq to respect the 12,000 Tokens Per Minute (TPM) free-tier ceiling.

### 3. API Quota Time-Gating & GitHub Actions Cron Alignment
* Gating protects SerpApi (250/mo limit) and Apify ($5/mo credits) by restricting their execution to a single morning run:
  * `0 2 * * *` (02:00 UTC = 07:30 AM IST): Executes **all 6 scrapers** (Morning Run).
  * `0 8 * * *` (08:00 UTC = 01:30 PM IST): Executes **free scrapers only** (Naukri, Adzuna, Workday).
  * `0 14 * * *` (14:00 UTC = 07:30 PM IST): Executes **free scrapers only** (Naukri, Adzuna, Workday).

### 4. Clustered Geographic Hub Rotation
* Prevents 30-second Google Jobs API timeouts while eliminating geographic blind spots by clustering locations into balanced rotation chunks:
  * **Regional Hubs:** Kerala, Kochi, Kozhikode, Thiruvananthapuram, Coimbatore.
  * **National Hubs:** Bengaluru, Pune, Hyderabad, Chennai, Mumbai, Delhi NCR.
  * **Global MedTech Hubs:** Minneapolis, Boston, San Diego, Galway (Ireland), Eindhoven (Netherlands), Singapore.

### 5. Multi-Layer Deduplication & Token Optimization
* **Layer 0 (O(1) Cloud Deduplication):** Checks Turso Cloud SQLite primary keys (`job_id`) and indexed URLs before downloading descriptions or invoking LLMs.
* **Layer 1 (Regex Spam Shield):** Word-boundary compiled regex strips irrelevant maintenance, technician, calibration, and sales roles without false positives.
* **Layer 2 (Token Optimization):** Strips HTML, scripts, and styling via BeautifulSoup; caps text at 3,500 characters to cut LLM inference latency in half.

### 6. DevSecOps & Security Hardening
* **Least-Privilege CI/CD:** GitHub Actions workflows explicitly declare `permissions: contents: read`.
* **Supply-Chain Pinning:** Third-party GitHub Actions (`actions/checkout`, `actions/setup-python`) pinned to immutable full commit SHAs.
* **Masked Telemetry:** Safe configuration reporting (`--check-config`) dynamically masks all credentials in run logs.

---

## 📁 Repository Structure

```text
Job Finder/
├── .github/
│   └── workflows/
│       └── main.yml               # Ephemeral scheduled runner on GitHub Actions
├── scrapers/                      # Modular native async scraping engines
│   ├── __init__.py                # Package collector & concurrent orchestration
│   ├── base.py                    # JobResult dataclass, dedup & normalization
│   ├── adzuna.py                  # Adzuna developer API scraper
│   ├── google_jobs.py             # SerpApi Google Jobs scraper with rotation
│   ├── linkedin.py                # LinkedIn Apify residential proxy scraper
│   ├── naukri.py                  # Playwright headless Chromium automation
│   ├── wellfound.py               # Apify Wellfound startup scraper
│   └── workday.py                 # Two-stage direct Workday ATS scraper
├── tests/                         # Comprehensive pytest test suite (111 tests)
│   ├── test_database.py           # Turso Cloud DB schema & persistence tests
│   ├── test_discord.py            # Discord embed formatting & SSRF tests
│   ├── test_evaluator.py          # Gemini & Groq fallback evaluation tests
│   ├── test_filters.py            # Regex spam & token optimization tests
│   ├── test_quota.py              # SerpApi location rotation & quota tests
│   └── test_scrapers.py           # Isolation, time-gating & parsing tests
├── config.py                      # Centralized configuration & secret masking
├── database.py                    # Turso Cloud SQLite async layer (turso-python)
├── discord_alerts.py              # Async Discord notification engine (aiohttp)
├── evaluator.py                   # Multi-provider LLM evaluator (Gemini + Groq)
├── filters.py                     # Layer 1 regex filter & Layer 2 token trimmer
├── main.py                        # Pipeline entrypoint, CLI & orchestration
├── quota_manager.py               # Location chunk rotation & tracking
├── pyproject.toml                 # Pytest configuration & tool settings
├── requirements.txt               # Pinned Python package dependencies
├── .env.example                   # Pristine environment variable template
└── .gitignore                     # Strict Git exclusion rules
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.11 or higher
* [Turso CLI](https://docs.turso.tech/cli/introduction) or Turso Cloud account
* API Keys: Google AI Studio (Gemini), Groq Cloud, SerpApi, Apify (optional), Adzuna (optional)
* Discord Webhook URL

### 2. Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/deeepak10/job-finder.git
cd job-finder

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install --with-deps chromium
```

### 3. Environment Configuration

Copy the template and configure your secrets:
```bash
cp .env.example .env
```

Edit `.env`:
```ini
# Primary LLM: Google Gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.6-flash

# Fallback LLM: Groq Llama-3.1-8B
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant

# Cloud Database: Turso (LibSQL)
TURSO_DATABASE_URL=libsql://your-database-name.turso.io
TURSO_AUTH_TOKEN=your_turso_auth_token

# Discord Notification Webhook
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Scraper Credentials
SERPAPI_API_KEY=your_serpapi_key
APIFY_TOKEN=your_apify_token
ADZUNA_APP_ID=your_adzuna_app_id
ADZUNA_APP_KEY=your_adzuna_app_key
```

### 4. Configuration & Security Audit

Verify that all credentials are valid and safe:
```bash
python main.py --check-config
```

---

## 💻 CLI Commands & Usage

```bash
# 1. Run the full autonomous pipeline
python main.py

# 2. Perform a dry-run (scrapes, filters, and evaluates without saving or alerting)
python main.py --dry-run

# 3. Force execution of quota-heavy scrapers regardless of current hour
python main.py --force-quota-scrapers

# 4. Run with selective scrapers bypassed
python main.py --no-naukri --no-google-jobs --no-workday

# 5. Query live Turso database metrics
python main.py --stats

# 6. Dispatch a sample high-priority Discord alert to test webhook
python main.py --test-alert

# 7. Enable detailed debug logging
python main.py -v
```

---

## 🧪 Automated Testing

The repository contains an exhaustive test suite covering scrapers, filters, database persistence, LLM waterfalls, and Discord notifications:

```bash
# Run all tests
pytest -v

# Run with test coverage
pytest --cov=.
```

**Test Suite Coverage (111 Passed Tests):**
* `test_scrapers.py`: Validates location classifications, JobResult schemas, URL builders, failure isolation, two-stage Workday scraping, and morning vs. off-peak time-gating.
* `test_evaluator.py`: Tests Gemini structured output, AFC disabling, 4.5s pacing, 503 retry mechanisms, Groq 6s pacing, markdown stripping, and seamless 429 fallback simulation.
* `test_database.py`: Tests Turso idempotent schema creation, O(1) deduplication, hashing algorithms, and row normalization.
* `test_filters.py`: Validates regex boundaries against false positives (e.g. permitting *"Cloud Service Architect"* while rejecting *"Field Service Engineer"*).
* `test_quota.py`: Tests clustered geographical rotation chunks and monthly SerpApi quota caps.
* `test_discord.py`: Validates embed fields, SSRF defenses, length limits, and purple priority coloring.

---

## 🌐 Production Deployment (GitHub Actions)

The pipeline is pre-configured to run autonomously in GitHub Actions using ephemeral Ubuntu runners.

1. In your GitHub repository, navigate to **Settings &rarr; Secrets and variables &rarr; Actions**.
2. Add your secrets:
   * `TURSO_DATABASE_URL`
   * `TURSO_AUTH_TOKEN`
   * `GEMINI_API_KEY`
   * `GROQ_API_KEY`
   * `DISCORD_WEBHOOK_URL`
   * `SERPAPI_API_KEY`
   * `APIFY_TOKEN`
   * `LINKEDIN_ACTOR_ID`
   * `WELLFOUND_ACTOR_ID`
   * `ADZUNA_APP_ID`
   * `ADZUNA_APP_KEY`
3. The pipeline will automatically trigger on the aligned schedule (07:30 AM, 01:30 PM, and 07:30 PM IST), or manually via **Actions &rarr; Run workflow**.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details. Built for autonomous, high-precision technical career acceleration.
