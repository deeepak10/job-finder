# Job Automation Pipeline - System Architecture & Build Instructions

## Project Objective
Build an automated Python pipeline to scrape biomedical R&D, IoT, and Health-Tech software roles from LinkedIn, Naukri, Indeed, and Wellfound. The system must filter out duplicates, check for visa sponsorships for international roles, and push real-time alerts to a Discord Webhook.

## Developer Instructions
You are acting as the Lead Developer. Follow these phases sequentially. Do not move to the next phase until the current one is tested and functional. 

### Phase 1: Environment & Database Setup
1. Initialize a Python virtual environment and create `requirements.txt`.
2. Create a SQLite database (`jobs.db`) with a table `job_postings`.
3. Columns must include: `job_id` (Primary Key), `title`, `company`, `location`, `platform`, `url`, `sponsorship_offered` (Boolean), `date_found`.
4. Write a helper function `is_new_job(job_id)` to verify if a job already exists before processing.

### Phase 2: Building the Scrapers (The 4 Target Platforms)
*Technical Constraint: Due to high anti-bot protection, strictly follow the scraping strategies below.*

1. **Naukri (India):** 
   * Use `playwright` (headless mode).
   * Target search URLs for "Biomedical Engineer R&D", "Python Healthtech", and "Medical IoT".
   * Implement random delays and wait for job cards to render to avoid blocks. Extract title, company, and application link.
2. **Wellfound (Health-Tech Startups):**
   * Use a job API wrapper (like Apify or RapidAPI startup job endpoints).
   * Filter specifically for "Health Tech" and "Medical Devices". Look for "Python", "Hardware", and "Data".
3. **LinkedIn & Indeed (Corporate / Global):**
   * Use the `google-search-results` (SerpApi) library, specifically the Google Jobs engine.
   * Query: `(Biomedical OR ECE OR Python) AND (R&D OR Data OR Hardware)`
   * Set up one function for India locations, and another for Europe, the UK, and Singapore.

### Phase 3: The Global Visa & Tech Keyword Filter
1. Write a function that processes the raw job description text.
2. **Tech Matcher:** Check for target stack keywords: `Python`, `Next.js`, `WebSockets`, `OpenCV`, `MediaPipe`, `IoT`, `Arduino`.
3. **Visa Matcher:** If the location is outside India, search the text for these exact phrases: `"visa sponsorship"`, `"relocation support"`, `"relocation assistance"`, `"tier 2 visa"`, `"Blue Card"`.
4. If a global job lacks visa keywords or explicitly states "no sponsorship", discard it immediately.

### Phase 4: Discord Webhook Integration
1. Create a function `send_discord_alert(job_dict, webhook_url)`.
2. Use Discord Rich Embeds to format the alert:
   * **Title:** Job Title (hyperlinked to the URL).
   * **Color:** Green if `sponsorship_offered == True`, Blue if India-based, Orange for startups.
   * **Fields:** Company, Location, Platform (Naukri/Wellfound/LinkedIn/Indeed).
   * **Custom Footer Logic:** 
     * If the job mentions "Computer Vision" or "Image Processing", append: *"💡 Tip: Highlight the Gesture Recognition project in your application!"*
     * If the job mentions "IoT", "Streaming", or "Web Apps", append: *"💡 Tip: Highlight the ECG Telemetry WebSockets project!"*

### Phase 5: Automation & Deployment
1. Create `main.py` to run the scrapers concurrently using `asyncio` or `concurrent.futures`.
2. Write a GitHub Actions `.yml` file to run `main.py` automatically every 4 hours.
3. Configure the script to read all sensitive credentials (SerpApi Key, Apify Token, Discord Webhook) from `.env` locally or GitHub Secrets in production.