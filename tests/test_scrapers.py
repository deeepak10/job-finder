"""Phase 2 tests — pure logic only, no network. Run: pytest -q"""

import pytest

import config
from scrapers.base import JobResult, dedupe, is_india, platform_from_url, valid_only
from scrapers import google_jobs, naukri, wellfound


# ---------------- location classification ----------------

@pytest.mark.parametrize("loc", [
    "Bengaluru, Karnataka, India",
    "Kochi, Kerala",
    "Remote - India",
    "Pune",
])
def test_indian_locations(loc):
    assert is_india(loc) is True


@pytest.mark.parametrize("loc", [
    "Berlin, Germany",
    "London, United Kingdom",
    "Singapore",
    "Indianapolis, Indiana, USA",   # substring trap
    "Delhi, Ohio, USA",             # namesake trap
    "",
])
def test_non_indian_locations(loc):
    assert is_india(loc) is False


def test_is_international_flag_is_derived():
    india = JobResult(title="A", company="B", url="https://x.com/1",
                      platform="naukri", location="Chennai, India")
    abroad = JobResult(title="A", company="B", url="https://x.com/2",
                       platform="indeed", location="Munich, Germany")
    assert india.is_international is False
    assert abroad.is_international is True


# ---------------- JobResult ----------------

def test_validity_and_cleaning():
    job = JobResult(title="  Medical   IoT\nEngineer ", company=" Acme ",
                    url="https://naukri.com/job/1", platform="Naukri")
    assert job.title == "Medical IoT Engineer"
    assert job.platform == "naukri"
    assert job.is_valid()
    assert JobResult(title="", company="Acme", url="https://x.com",
                     platform="indeed").is_valid() is False


def test_to_row_matches_database_contract():
    job = JobResult(title="R&D Engineer", company="EuroMed", location="Berlin, Germany",
                    url="https://linkedin.com/jobs/view/1", platform="linkedin")
    row = job.to_row()
    assert row["title"] == "R&D Engineer"
    assert row["company"] == "EuroMed"
    assert row["location"] == "Berlin, Germany"
    assert row["platform"] == "linkedin"
    assert row["url"].startswith("https://")
    assert "date_found" in row



def test_dedupe_within_run():
    a = JobResult(title="X", company="Y", url="https://indeed.com/viewjob?jk=1",
                  platform="indeed")
    b = JobResult(title="X", company="Y", url="https://indeed.com/viewjob?jk=1&from=alert",
                  platform="indeed")
    assert len(dedupe([a, b])) == 1


def test_valid_only_drops_malformed():
    good = JobResult(title="X", company="Y", url="https://x.com/1", platform="naukri")
    bad = JobResult(title="X", company="Y", url="javascript:void(0)", platform="naukri")
    assert valid_only([good, bad]) == [good]


def test_platform_from_url():
    assert platform_from_url("https://www.linkedin.com/jobs/view/12") == "linkedin"
    assert platform_from_url("https://in.indeed.com/viewjob?jk=9") == "indeed"
    assert platform_from_url("https://example.com/x", default="naukri") == "naukri"


# ---------------- Google Jobs / SerpApi parsing ----------------

SERP_RESULT = {
    "title": "Biomedical R&D Engineer",
    "company_name": "Nova Devices",
    "location": "Munich, Germany",
    "description": "Python, hardware. Visa sponsorship available.",
    "detected_extensions": {"posted_at": "2 days ago"},
    "apply_options": [
        {"title": "Apply on Company Site", "link": "https://nova.example/careers/1"},
        {"title": "Apply on LinkedIn", "link": "https://www.linkedin.com/jobs/view/1"},
    ],
}


def test_serp_picks_targeted_board_not_first_option():
    job = google_jobs._to_job(SERP_RESULT)
    assert job.platform == "linkedin"
    assert "linkedin.com" in job.url
    assert job.is_international is True


def test_serp_skips_untracked_boards():
    result = dict(SERP_RESULT, apply_options=[
        {"title": "Apply on Glassdoor", "link": "https://glassdoor.com/j/1"}])
    assert google_jobs._to_job(result) is None


def test_serp_falls_back_to_related_links():
    result = dict(SERP_RESULT)
    result.pop("apply_options")
    result["related_links"] = [{"link": "https://in.indeed.com/viewjob?jk=77"}]
    job = google_jobs._to_job(result)
    assert job.platform == "indeed"


def test_serp_requires_api_key(monkeypatch):
    monkeypatch.setattr(config, "SERPAPI_API_KEY", "")
    with pytest.raises(google_jobs.MissingCredentials):
        google_jobs._search("q", "Singapore")


# ---------------- Wellfound normalization ----------------

def test_wellfound_handles_alternate_field_names():
    item = {"jobTitle": "Firmware Engineer", "startupName": "CardioLabs",
            "locationNames": ["Remote", "Singapore"], "jobUrl": "/jobs/555-firmware",
            "descriptionText": "Embedded C, Python, medical devices", "id": "555"}
    job = wellfound._normalize(item)
    assert job.title == "Firmware Engineer"
    assert job.company == "CardioLabs"
    assert job.location == "Remote, Singapore"
    assert job.url == "https://wellfound.com/jobs/555-firmware"
    assert job.job_id == "wf-555"      # native id, not URL-derived


def test_wellfound_focus_filter():
    on = wellfound._normalize({"title": "ML Engineer", "company": "MedAI",
                               "url": "https://wellfound.com/jobs/1",
                               "description": "clinical imaging, Python"})
    off = wellfound._normalize({"title": "Growth Marketer", "company": "Shoply",
                                "url": "https://wellfound.com/jobs/2",
                                "description": "run paid ads"})
    assert wellfound.matches_focus(on) is True
    assert wellfound.matches_focus(off) is False


def test_wellfound_schema_mapping_orgupdate():
    item = {
        "job_title": "Biomedical R&D Engineer",
        "company_name": "NeuroTech Systems",
        "URL": "https://wellfound.com/jobs/99-bio-rd",
        "location": "Boston, MA",
        "description": "Medical IoT sensors",
    }
    job = wellfound._normalize(item)
    assert job is not None
    assert job.title == "Biomedical R&D Engineer"
    assert job.company == "NeuroTech Systems"
    assert job.url == "https://wellfound.com/jobs/99-bio-rd"
    assert job.location == "Boston, MA"


def test_wellfound_schema_mapping_location_default():
    # When location is missing or None, it must default to "Not specified"
    item1 = {
        "job_title": "AI Clinical Specialist",
        "company_name": "HealthVision",
        "URL": "https://wellfound.com/jobs/101",
    }
    job1 = wellfound._normalize(item1)
    assert job1 is not None
    assert job1.location == "Not specified"

    item2 = {
        "title": "AI Clinical Specialist",
        "company": "HealthVision",
        "url": "https://wellfound.com/jobs/102",
        "location": None,
    }
    job2 = wellfound._normalize(item2)
    assert job2 is not None
    assert job2.location == "Not specified"


def test_wellfound_schema_mapping_skips_none_fields():
    # Skip if title is None
    assert wellfound._normalize({"job_title": None, "title": None, "company": "MedTech", "URL": "https://wellfound.com/jobs/1"}) is None
    # Skip if company is None
    assert wellfound._normalize({"job_title": "Engineer", "company_name": None, "company": None, "URL": "https://wellfound.com/jobs/1"}) is None
    # Skip if url is None
    assert wellfound._normalize({"job_title": "Engineer", "company_name": "MedTech", "URL": None, "url": None}) is None


def test_wellfound_requires_credentials(monkeypatch):
    monkeypatch.setattr(config, "APIFY_TOKEN", "")
    with pytest.raises(wellfound.MissingCredentials):
        wellfound._run_actor()


# ---------------- Naukri URL building ----------------

def test_naukri_url_building():
    first = naukri.build_search_url("medical-iot-jobs", "medical IoT", 1)
    second = naukri.build_search_url("medical-iot-jobs", "medical IoT", 2)
    assert first == "https://www.naukri.com/medical-iot-jobs?k=medical+IoT"
    assert "medical-iot-jobs-2" in second


# ---------------- Scraper Query Optimization Tests ----------------

EXPECTED_TARGET_QUERIES = [
    "Medical Device R&D",
    "Biomedical Firmware",
    "Medical IoT",
    "IoT Medical Devices",
    "Python Healthtech",
    "Signal Processing Engineer",
    "R&D Engineer Medical",
]


def test_target_queries_configured():
    assert hasattr(config, "TARGET_QUERIES")
    assert config.TARGET_QUERIES == EXPECTED_TARGET_QUERIES

    # Generic degree-based queries MUST be removed
    for q in config.TARGET_QUERIES:
        assert q not in ("Biomedical Engineer", "Biomedical Engineering")

    # Check Naukri searches
    naukri_keywords = [k for _, k in config.NAUKRI_SEARCHES]
    for exp in EXPECTED_TARGET_QUERIES:
        assert exp in naukri_keywords
    for slug, kw in config.NAUKRI_SEARCHES:
        assert kw not in ("biomedical engineer R&D", "Biomedical Engineer", "Biomedical Engineering")
        assert slug != "biomedical-engineer-rnd-jobs"

    # Check Google Jobs queries
    assert config.GOOGLE_JOBS_QUERIES == EXPECTED_TARGET_QUERIES
    assert "Biomedical Engineer" not in config.GOOGLE_JOBS_QUERIES

    # Check Wellfound configuration
    assert config.WELLFOUND_INDUSTRY_TAGS == ["Health Tech", "Medical Devices"]
    assert config.WELLFOUND_KEYWORDS == EXPECTED_TARGET_QUERIES


def test_locations_remain_untouched():
    # India locations
    assert len(config.INDIA_LOCATIONS) >= 4
    assert any("Bengaluru" in loc for loc in config.INDIA_LOCATIONS)
    assert any("Hyderabad" in loc for loc in config.INDIA_LOCATIONS)

    # Global locations: EU, UK, Singapore
    assert any("Germany" in loc for loc in config.GLOBAL_LOCATIONS)
    assert any("United Kingdom" in loc for loc in config.GLOBAL_LOCATIONS)
    assert any("Singapore" in loc for loc in config.GLOBAL_LOCATIONS)


def test_serp_loops_all_queries(monkeypatch):
    searched_queries = []
    searched_locations = []

    def mock_search(query: str, location: str, page: int = 0):
        searched_queries.append(query)
        searched_locations.append(location)
        return []

    monkeypatch.setattr(google_jobs, "_search", mock_search)
    google_jobs._scrape_locations(["Singapore", "Germany"], "test")

    # All target queries should have been queried
    assert set(searched_queries) == set(EXPECTED_TARGET_QUERIES)
    # Total search calls should be len(queries) * len(locations)
    assert len(searched_queries) == len(EXPECTED_TARGET_QUERIES) * 2


def test_wellfound_actor_payload(monkeypatch):
    from datetime import timedelta
    recorded_input = {}
    recorded_kwargs = {}

    class MockRun:
        default_dataset_id = "ds_123"

    class MockActor:
        def call(self, run_input=None, **kwargs):
            nonlocal recorded_input, recorded_kwargs
            recorded_input = run_input
            recorded_kwargs = kwargs
            return MockRun()

    class MockDataset:
        def iterate_items(self):
            return []

    class MockClient:
        def __init__(self, token):
            pass

        def actor(self, actor_id):
            return MockActor()

        def dataset(self, ds_id):
            return MockDataset()

    monkeypatch.setattr(config, "APIFY_TOKEN", "mock-token")
    monkeypatch.setattr(config, "WELLFOUND_ACTOR_ID", "mock-actor-id")
    monkeypatch.setattr("apify_client.ApifyClient", MockClient)

    wellfound._run_actor()

    assert recorded_input["search"] == "Health Tech"
    assert "India" in recorded_input["locations"]
    assert "United States" in recorded_input["locations"]
    assert "Health Tech" in recorded_input["industryTags"]
    assert "Medical Devices" in recorded_input["industryTags"]
    assert recorded_input["keywords"] == EXPECTED_TARGET_QUERIES
    assert recorded_input["maxItems"] == 20
    assert recorded_kwargs.get("wait_duration") == timedelta(seconds=90)
    assert recorded_kwargs.get("run_timeout") == timedelta(seconds=90)


def test_wellfound_actor_dict_fallback(monkeypatch):
    class MockActor:
        def call(self, run_input=None, **kwargs):
            return {"defaultDatasetId": "ds_legacy"}

    class MockDataset:
        def iterate_items(self):
            return [{"title": "Bio Engineer", "company": "MedCo"}]

    class MockClient:
        def __init__(self, token):
            pass

        def actor(self, actor_id):
            return MockActor()

        def dataset(self, ds_id):
            assert ds_id == "ds_legacy"
            return MockDataset()

    monkeypatch.setattr(config, "APIFY_TOKEN", "mock-token")
    monkeypatch.setattr(config, "WELLFOUND_ACTOR_ID", "mock-actor-id")
    monkeypatch.setattr("apify_client.ApifyClient", MockClient)

    items = wellfound._run_actor()
    assert len(items) == 1
    assert items[0]["title"] == "Bio Engineer"


def test_wellfound_scrape_async_timeout(monkeypatch):
    import asyncio
    called_timeout = None

    async def mock_wait_for(fut, timeout):
        nonlocal called_timeout
        called_timeout = timeout
        fut.close()
        raise asyncio.TimeoutError()

    monkeypatch.setattr("asyncio.wait_for", mock_wait_for)
    result = asyncio.run(wellfound.scrape_async())
    assert result == []
    assert called_timeout == 90.0


def test_scraper_depth_and_pagination_limits():
    """Verify that scrapers are strictly capped to the freshest 1 page / 20 results."""
    assert config.NAUKRI_MAX_PAGES == 1
    assert config.NAUKRI_TIMEOUT_MS == 30_000
    assert config.WELLFOUND_MAX_RESULTS == 20
    assert getattr(config, "SERPAPI_MAX_PAGES", 1) == 1
    assert getattr(config, "SERPAPI_MAX_RESULTS_PER_QUERY", 20) == 20
    assert getattr(config, "GEMINI_TIMEOUT_SECONDS", 45.0) == 45.0


def test_serp_caps_results_per_query(monkeypatch):
    """Ensure SerpApi scraper caps results at 20 per query even if API returns more."""
    # Generate 30 fake results
    mock_results = [
        {
            "title": f"Job {i}",
            "company_name": f"Company {i}",
            "location": "Bengaluru, India",
            "apply_options": [{"title": "Apply on LinkedIn", "link": f"https://www.linkedin.com/jobs/view/{i}"}],
        }
        for i in range(30)
    ]

    monkeypatch.setattr(google_jobs, "_search", lambda q, loc, page=0: mock_results)
    monkeypatch.setattr(google_jobs, "sleep_jitter", lambda lo, hi: None)
    monkeypatch.setattr(config, "GOOGLE_JOBS_QUERIES", ["Medical IoT"])

    jobs = google_jobs._scrape_locations(["Bengaluru, Karnataka, India"], "test")
    # Must cap to at most 20 results
    assert len(jobs) <= 20


def test_pipeline_expansion_config():
    """Verify scraper toggles and SERPAPI_LOCATIONS for high-yield pipeline expansion."""
    assert config.NAUKRI_ENABLED is True
    assert config.GOOGLE_JOBS_ENABLED is True
    assert config.WELLFOUND_ENABLED is True
    assert config.LINKEDIN_ENABLED is True
    assert config.ADZUNA_ENABLED is True
    assert config.WORKDAY_ENABLED is True
    assert "Bengaluru, Karnataka, India" in config.SERPAPI_LOCATIONS
    assert "Minneapolis, MN" in config.SERPAPI_LOCATIONS
    assert "Eindhoven, Netherlands" in config.SERPAPI_LOCATIONS
    assert "Kochi, Kerala, India" in config.SERPAPI_LOCATIONS
    assert len(config.REGIONAL_HUBS) == 5
    assert len(config.NATIONAL_HUBS) == 4
    assert len(config.GLOBAL_HUBS) == 5
    assert "adzuna" in config.PLATFORMS
    assert "workday" in config.PLATFORMS


def test_naukri_has_scrape_async():
    """Ensure naukri module exposes scrape_async."""
    assert hasattr(naukri, "scrape_async")
    assert callable(naukri.scrape_async)


def test_adzuna_skips_without_credentials(monkeypatch):
    from scrapers import adzuna
    monkeypatch.delenv("ADZUNA_APP_ID", raising=False)
    monkeypatch.delenv("ADZUNA_APP_KEY", raising=False)
    import asyncio
    jobs = asyncio.run(adzuna.scrape_async())
    assert jobs == []


def test_adzuna_parses_success_and_handles_errors(monkeypatch):
    import asyncio
    from scrapers import adzuna
    monkeypatch.setenv("ADZUNA_APP_ID", "test-app-id")
    monkeypatch.setenv("ADZUNA_APP_KEY", "test-app-key")

    mock_response_data = {
        "results": [
            {
                "title": "Embedded Firmware Engineer",
                "redirect_url": "https://www.adzuna.in/land/ad/123",
                "company": {"display_name": "HealthTech Labs"},
                "location": {"display_name": "Bengaluru, India"},
            }
        ]
    }

    class MockResponse:
        def __init__(self, status, data):
            self.status = status
            self._data = data

        async def json(self):
            return self._data

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockSession:
        def __init__(self, status=200, data=None, raise_exc=None):
            self.status = status
            self.data = data or {}
            self.raise_exc = raise_exc

        def get(self, url, timeout=None):
            if self.raise_exc:
                raise self.raise_exc
            return MockResponse(self.status, self.data)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    # Test success 200
    monkeypatch.setattr("aiohttp.ClientSession", lambda: MockSession(200, mock_response_data))
    jobs = asyncio.run(adzuna.scrape_async())
    assert len(jobs) == 1
    assert jobs[0]["title"] == "Embedded Firmware Engineer"
    assert jobs[0]["company"] == "HealthTech Labs"
    assert jobs[0]["source"] == "adzuna"

    # Test error status (fail-open on 429 and 503)
    monkeypatch.setattr("aiohttp.ClientSession", lambda: MockSession(429, {}))
    jobs = asyncio.run(adzuna.scrape_async())
    assert jobs == []

    monkeypatch.setattr("aiohttp.ClientSession", lambda: MockSession(503, {}))
    jobs = asyncio.run(adzuna.scrape_async())
    assert jobs == []

    # Test network exception (fail-open)
    monkeypatch.setattr("aiohttp.ClientSession", lambda: MockSession(raise_exc=RuntimeError("Connection reset")))
    jobs = asyncio.run(adzuna.scrape_async())
    assert jobs == []


def test_linkedin_scraper_fail_open(monkeypatch):
    import asyncio
    from scrapers import linkedin

    # Apify mode fails safely
    monkeypatch.setenv("LINKEDIN_ACTOR_ID", "mock-actor")
    monkeypatch.setenv("APIFY_TOKEN", "mock-token")

    async def mock_apify_fail(*args, **kwargs):
        raise RuntimeError("Apify rate limit")

    monkeypatch.setattr(linkedin, "_scrape_via_apify", mock_apify_fail)
    jobs = asyncio.run(linkedin.scrape_async())
    assert jobs == []

    # JobSpy mode fails safely on error / missing module
    monkeypatch.delenv("LINKEDIN_ACTOR_ID", raising=False)
    monkeypatch.delenv("APIFY_TOKEN", raising=False)

    def mock_jobspy_fail():
        raise RuntimeError("Cloud IP block")

    monkeypatch.setattr(linkedin, "_run_jobspy_sync", mock_jobspy_fail)
    jobs = asyncio.run(linkedin.scrape_async())
    assert jobs == []


def test_linkedin_apify_payload(monkeypatch):
    import asyncio
    from scrapers import linkedin

    recorded_input = {}

    class MockActor:
        async def call(self, run_input=None, **kwargs):
            nonlocal recorded_input
            recorded_input = run_input
            return {"defaultDatasetId": "ds_test"}

    class MockDataset:
        async def iterate_items(self):
            if False:
                yield {}

    class MockClient:
        def __init__(self, token):
            pass

        def actor(self, actor_id):
            return MockActor()

        def dataset(self, ds_id):
            return MockDataset()

    monkeypatch.setattr("apify_client.ApifyClientAsync", MockClient)
    jobs = asyncio.run(linkedin._scrape_via_apify("test-actor", "test-token"))
    assert recorded_input["queries"] == "embedded firmware"
    assert recorded_input["keywords"] == "embedded firmware"
    assert recorded_input["jobTitle"] == "embedded firmware"
    assert recorded_input["title"] == "embedded firmware"
    assert recorded_input["location"] == "India"
    assert recorded_input["maxItems"] == 20


def test_linkedin_location_worldwide_fallback(monkeypatch):
    import asyncio
    from scrapers import linkedin

    class MockActor:
        async def call(self, run_input=None, **kwargs):
            return {"defaultDatasetId": "ds_test"}

    class MockDataset:
        async def iterate_items(self):
            yield {"title": "Firmware Engineer", "company": "MedCo", "url": "https://linkedin.com/jobs/1"}

    class MockClient:
        def __init__(self, token):
            pass

        def actor(self, actor_id):
            return MockActor()

        def dataset(self, ds_id):
            return MockDataset()

    monkeypatch.setattr("apify_client.ApifyClientAsync", MockClient)
    jobs = asyncio.run(linkedin._scrape_via_apify("test-actor", "test-token"))
    assert len(jobs) == 1
    assert jobs[0]["location"] == "Worldwide"


def test_scraper_isolation_fail_open():
    import asyncio
    from scrapers.base import JobResult

    async def good_scraper():
        return [JobResult("Firmware Eng", "Skanray", "https://example.com/1", "naukri")]

    async def crashing_scraper():
        raise RuntimeError("Cloud IP block or Rate limit 429")

    async def dict_scraper():
        return [{"title": "Bio Eng", "company": "MedTech", "url": "https://example.com/2", "source": "adzuna"}]

    async def run_gather():
        results = await asyncio.gather(
            good_scraper(),
            crashing_scraper(),
            dict_scraper(),
            return_exceptions=True,
        )
        all_jobs = []
        for res in results:
            if isinstance(res, list):
                for item in res:
                    if isinstance(item, JobResult):
                        all_jobs.append(item)
                    elif isinstance(item, dict):
                        all_jobs.append(JobResult(
                            title=item.get("title", ""),
                            company=item.get("company", ""),
                            url=item.get("url", ""),
                            platform=item.get("source", "adzuna"),
                        ))
        return all_jobs

    jobs = asyncio.run(run_gather())
    assert len(jobs) == 2
    assert jobs[0].title == "Firmware Eng"
    assert jobs[1].title == "Bio Eng"
    assert jobs[1].platform == "adzuna"


def test_workday_scraper_success(monkeypatch):
    """Verify workday.scrape_async extracts job postings and descriptions from mock responses."""
    import asyncio
    from scrapers import workday

    async def mock_sleep(*args, **kwargs):
        pass

    monkeypatch.setattr("asyncio.sleep", mock_sleep)

    class MockPostResponse:
        status = 200
        headers = {"Content-Type": "application/json"}
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def json(self):
            return {
                "jobPostings": [
                    {
                        "title": "Principal Firmware Engineer",
                        "externalPath": "/job/Minneapolis-MN/Principal-Firmware-Engineer_R1020",
                        "locationsText": "Minneapolis, MN, United States",
                    }
                ]
            }

    class MockGetResponse:
        status = 200
        headers = {"Content-Type": "application/json"}
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def json(self):
            return {
                "jobPostingInfo": {
                    "jobDescription": "<p>Design firmware for implantable medical devices.</p>"
                }
            }

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def post(self, url, json=None, headers=None):
            return MockPostResponse()
        def get(self, url, headers=None):
            return MockGetResponse()

    monkeypatch.setattr("aiohttp.ClientSession", MockSession)

    jobs = asyncio.run(workday.scrape_async())
    # 10 tenants, 1 unique posting each (deduped across keywords)
    assert len(jobs) == len(workday.MEDTECH_WORKDAY_TENANTS)
    assert jobs[0]["title"] == "Principal Firmware Engineer"
    assert jobs[0]["company"] == "Medtronic"
    assert "medtronic.wd1.myworkdayjobs.com/en-US/External/job/" in jobs[0]["url"]
    assert jobs[0]["location"] == "Minneapolis, MN, United States"
    assert jobs[0]["description"] == "<p>Design firmware for implantable medical devices.</p>"
    assert jobs[0]["source"] == "Workday ATS"
    assert jobs[0]["tier"] == "strict"


def test_workday_scraper_handles_network_error(monkeypatch):
    """Verify workday.scrape_async handles network exceptions safely."""
    import asyncio
    from scrapers import workday

    async def mock_sleep(*args, **kwargs):
        pass

    monkeypatch.setattr("asyncio.sleep", mock_sleep)

    class MockCrashingSession:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def post(self, url, json=None, headers=None):
            raise RuntimeError("Connection refused")

    monkeypatch.setattr("aiohttp.ClientSession", MockCrashingSession)

    jobs = asyncio.run(workday.scrape_async())
    assert jobs == []


def test_workday_scraper_handles_html_maintenance(monkeypatch):
    """Verify workday.scrape_async gracefully ignores HTML maintenance responses."""
    import asyncio
    from scrapers import workday

    async def mock_sleep(*args, **kwargs):
        pass

    monkeypatch.setattr("asyncio.sleep", mock_sleep)

    class MockHtmlResponse:
        status = 200
        headers = {"Content-Type": "text/html; charset=utf-8"}
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def json(self):
            raise ValueError("Unexpected mimetype: text/html")

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def post(self, url, json=None, headers=None):
            return MockHtmlResponse()

    monkeypatch.setattr("aiohttp.ClientSession", MockSession)
    jobs = asyncio.run(workday.scrape_async())
    assert jobs == []


def test_workday_scraper_handles_rate_limit_429(monkeypatch):
    """Verify workday.scrape_async handles 429 rate limit by breaking keyword loop for tenant."""
    import asyncio
    from scrapers import workday

    async def mock_sleep(*args, **kwargs):
        pass

    monkeypatch.setattr("asyncio.sleep", mock_sleep)

    class Mock429Response:
        status = 429
        headers = {"Content-Type": "application/json"}
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def json(self):
            return {"error": "Rate limit exceeded"}

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def post(self, url, json=None, headers=None):
            return Mock429Response()

    monkeypatch.setattr("aiohttp.ClientSession", MockSession)
    jobs = asyncio.run(workday.scrape_async())
    assert jobs == []


def test_fetch_workday_jobs_helper(monkeypatch):
    """Verify fetch_workday_jobs checks Content-Type header before parsing JSON."""
    from scrapers import workday

    class MockResponse:
        def __init__(self, status_code, content_type, data):
            self.status_code = status_code
            self.headers = {"Content-Type": content_type}
            self._data = data
        def json(self):
            return self._data

    # Valid JSON
    monkeypatch.setattr("requests.get", lambda url: MockResponse(200, "application/json", [{"id": "1"}]))
    assert workday.fetch_workday_jobs("https://api.workday.com") == [{"id": "1"}]

    # HTML Maintenance page
    monkeypatch.setattr("requests.get", lambda url: MockResponse(200, "text/html", "<html>Maintenance</html>"))
    assert workday.fetch_workday_jobs("https://api.workday.com") == []


def test_run_scrapers_concurrently_morning_run(monkeypatch):
    """At or before 04:00 UTC (morning run), all scrapers including quota-heavy ones are executed."""
    import argparse
    import asyncio
    from main import run_scrapers_concurrently
    from scrapers.base import JobResult

    called = []

    async def mock_naukri():
        called.append("naukri")
        return [JobResult(title="Dev", company="C1", url="https://naukri.com/1", platform="naukri")]

    async def mock_adzuna():
        called.append("adzuna")
        return [JobResult(title="Dev", company="C2", url="https://adzuna.com/1", platform="adzuna")]

    async def mock_workday():
        called.append("workday")
        return [JobResult(title="Dev", company="C3", url="https://workday.com/1", platform="workday")]

    async def mock_google():
        called.append("google_jobs")
        return [JobResult(title="Dev", company="C4", url="https://google.com/1", platform="google_jobs")]

    async def mock_linkedin():
        called.append("linkedin")
        return [JobResult(title="Dev", company="C5", url="https://linkedin.com/1", platform="linkedin")]

    async def mock_wellfound():
        called.append("wellfound")
        return [JobResult(title="Dev", company="C6", url="https://wellfound.com/1", platform="wellfound")]

    monkeypatch.setattr("scrapers.naukri.scrape_async", mock_naukri)
    monkeypatch.setattr("scrapers.adzuna.scrape_async", mock_adzuna)
    monkeypatch.setattr("scrapers.workday.scrape_async", mock_workday)
    monkeypatch.setattr("scrapers.google_jobs.scrape_async", mock_google)
    monkeypatch.setattr("scrapers.linkedin.scrape_async", mock_linkedin)
    monkeypatch.setattr("scrapers.wellfound.scrape_async", mock_wellfound)

    args = argparse.Namespace(
        force_quota_scrapers=False,
        no_naukri=False,
        no_adzuna=False,
        no_workday=False,
        no_google_jobs=False,
        no_linkedin=False,
        no_wellfound=False,
    )

    # 02:00 UTC is 07:30 AM IST (morning run)
    jobs = asyncio.run(run_scrapers_concurrently(args=args, current_hour_utc=2))

    assert "naukri" in called
    assert "adzuna" in called
    assert "workday" in called
    assert "google_jobs" in called
    assert "linkedin" in called
    assert "wellfound" in called
    assert len(jobs) == 6


def test_run_scrapers_concurrently_offpeak_run(monkeypatch):
    """After 04:00 UTC (e.g. 08:00 UTC or 14:00 UTC), quota-heavy scrapers are skipped."""
    import argparse
    import asyncio
    from main import run_scrapers_concurrently
    from scrapers.base import JobResult

    called = []

    async def mock_naukri():
        called.append("naukri")
        return [JobResult(title="Dev", company="C1", url="https://naukri.com/1", platform="naukri")]

    async def mock_adzuna():
        called.append("adzuna")
        return [JobResult(title="Dev", company="C2", url="https://adzuna.com/1", platform="adzuna")]

    async def mock_workday():
        called.append("workday")
        return [JobResult(title="Dev", company="C3", url="https://workday.com/1", platform="workday")]

    async def mock_google():
        called.append("google_jobs")
        return []

    async def mock_linkedin():
        called.append("linkedin")
        return []

    async def mock_wellfound():
        called.append("wellfound")
        return []

    monkeypatch.setattr("scrapers.naukri.scrape_async", mock_naukri)
    monkeypatch.setattr("scrapers.adzuna.scrape_async", mock_adzuna)
    monkeypatch.setattr("scrapers.workday.scrape_async", mock_workday)
    monkeypatch.setattr("scrapers.google_jobs.scrape_async", mock_google)
    monkeypatch.setattr("scrapers.linkedin.scrape_async", mock_linkedin)
    monkeypatch.setattr("scrapers.wellfound.scrape_async", mock_wellfound)

    args = argparse.Namespace(
        force_quota_scrapers=False,
        no_naukri=False,
        no_adzuna=False,
        no_workday=False,
        no_google_jobs=False,
        no_linkedin=False,
        no_wellfound=False,
    )

    # 08:00 UTC (01:30 PM IST) and 14:00 UTC (07:30 PM IST) are off-peak
    jobs = asyncio.run(run_scrapers_concurrently(args=args, current_hour_utc=8))

    assert "naukri" in called
    assert "adzuna" in called
    assert "workday" in called
    assert "google_jobs" not in called
    assert "linkedin" not in called
    assert "wellfound" not in called
    assert len(jobs) == 3


def test_run_scrapers_concurrently_forced_during_offpeak(monkeypatch):
    """When force_quota_scrapers is True, quota scrapers execute even during off-peak hours."""
    import argparse
    import asyncio
    from main import run_scrapers_concurrently
    from scrapers.base import JobResult

    called = []

    async def mock_naukri():
        called.append("naukri")
        return []

    async def mock_adzuna():
        called.append("adzuna")
        return []

    async def mock_workday():
        called.append("workday")
        return []

    async def mock_google():
        called.append("google_jobs")
        return []

    async def mock_linkedin():
        called.append("linkedin")
        return []

    async def mock_wellfound():
        called.append("wellfound")
        return []

    monkeypatch.setattr("scrapers.naukri.scrape_async", mock_naukri)
    monkeypatch.setattr("scrapers.adzuna.scrape_async", mock_adzuna)
    monkeypatch.setattr("scrapers.workday.scrape_async", mock_workday)
    monkeypatch.setattr("scrapers.google_jobs.scrape_async", mock_google)
    monkeypatch.setattr("scrapers.linkedin.scrape_async", mock_linkedin)
    monkeypatch.setattr("scrapers.wellfound.scrape_async", mock_wellfound)

    args = argparse.Namespace(
        force_quota_scrapers=True,
        no_naukri=False,
        no_adzuna=False,
        no_workday=False,
        no_google_jobs=False,
        no_linkedin=False,
        no_wellfound=False,
    )

    asyncio.run(run_scrapers_concurrently(args=args, current_hour_utc=14))

    assert "naukri" in called
    assert "adzuna" in called
    assert "workday" in called
    assert "google_jobs" in called
    assert "linkedin" in called
    assert "wellfound" in called






