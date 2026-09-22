"""Tests for Turso Cloud Database layer."""

from database import (
    make_job_id,
    parse_turso_rows,
    title_company_hash,
)


def test_title_company_hash_is_deterministic_and_normalized():
    h1 = title_company_hash("Biomedical Engineer R&D", "GE HealthCare")
    h2 = title_company_hash("  biomedical engineer r&d  ", "ge healthcare")
    h3 = title_company_hash("Software Developer", "GE HealthCare")

    assert len(h1) == 16
    assert h1 == h2
    assert h1 != h3


def test_make_job_id_strips_url_tracking():
    id1 = make_job_id("naukri", "https://naukri.com/job/123?utm_source=alert&ref=feed", "Bio Eng", "Stryker")
    id2 = make_job_id("naukri", "https://naukri.com/job/123#frag", "Bio Eng", "Stryker")
    assert id1 == id2


def test_parse_turso_rows_extracts_nulls_and_values():
    raw_response = {
        "results": [
            {
                "type": "ok",
                "response": {
                    "type": "execute",
                    "result": {
                        "cols": [{"name": "id"}, {"name": "score"}, {"name": "missing"}],
                        "rows": [
                            [
                                {"type": "text", "value": "job_01"},
                                {"type": "integer", "value": "95"},
                                {"type": "null"},
                            ]
                        ],
                    },
                },
            }
        ]
    }
    parsed = parse_turso_rows(raw_response)
    assert len(parsed) == 1
    assert parsed[0]["id"] == "job_01"
    assert parsed[0]["score"] == "95"
    assert parsed[0]["missing"] is None


def test_schema_does_not_contain_removed_fields():
    from database import SCHEMA
    assert "ai_reasoning" not in SCHEMA
    assert "portfolio_highlight" not in SCHEMA
    assert "outreach_message" not in SCHEMA
    assert "visa_sponsorship" in SCHEMA


def test_add_job_sql_matches_lean_schema():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from database import add_job

    mock_client = MagicMock()
    mock_client.execute_query = AsyncMock(
        return_value={"results": [{"response": {"result": {"affected_row_count": 1}}}]}
    )
    mock_client.session = None

    job_dict = {
        "title": "Bio Eng",
        "company": "MedTech",
        "platform": "linkedin",
        "url": "https://example.com/job",
        "location": "Bengaluru",
        "visa_sponsorship": "Supported",
        # Legacy fields if present in dictionary should not be written to DB
        "ai_reasoning": "Some reasoning",
        "portfolio_highlight": "Project A",
        "outreach_message": "Draft",
        "job_category": "Biomedical_RD",
    }

    success = asyncio.run(add_job(job_dict, client=mock_client))
    assert success is True
    mock_client.execute_query.assert_called_once()
    call_args = mock_client.execute_query.call_args[0]
    sql, args = call_args[0], call_args[1]
    assert "ai_reasoning" not in sql
    assert "portfolio_highlight" not in sql
    assert "outreach_message" not in sql
    assert "job_category" in sql
    assert "desc_hash" in sql
    assert len(args) == 16  # job_id, title, company, location, platform, url, description, ai_score, visa_sponsorship, date_found, alert_sent, status, tier, match_reason, job_category, desc_hash
    assert "Some reasoning" not in args
    assert "Project A" not in args
    assert "Draft" not in args
    assert args[-2] == "Biomedical_RD"


def test_garbage_collection_nullifies_old_rejected_jobs():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from database import run_garbage_collection_async

    mock_client = MagicMock()
    mock_client.execute_query = AsyncMock(
        return_value={"results": [{"response": {"result": {"affected_row_count": 5}}}]}
    )
    mock_client.session = None

    cleared = asyncio.run(run_garbage_collection_async(client=mock_client))
    assert cleared == 5
    mock_client.execute_query.assert_called_once()
    sql = mock_client.execute_query.call_args[0][0]
    assert "UPDATE job_postings" in sql
    assert "SET description = NULL, match_reason = NULL" in sql
    assert "status = 'rejected'" in sql
    assert "date_found < datetime('now', '-30 days')" in sql


def test_add_job_parameter_alignment_and_category_fallbacks():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from database import add_job

    mock_client = MagicMock()
    mock_client.execute_query = AsyncMock(
        return_value={"results": [{"response": {"result": {"affected_row_count": 1}}}]}
    )
    mock_client.session = None

    # Job with missing/None optional fields and non-string category
    minimal_job = {
        "title": "Firmware Engineer",
        "company": "MedTech Devices",
        "platform": "naukri",
        "url": "https://naukri.com/firmware",
        "job_category": None,
    }

    success = asyncio.run(add_job(minimal_job, client=mock_client))
    assert success is True
    sql, args = mock_client.execute_query.call_args[0][0], mock_client.execute_query.call_args[0][1]
    assert len(args) == 16
    assert args[-2] == "General"  # Fallback from None to 'General'
    assert args[3] == "Unknown"   # Location fallback


def test_generate_desc_hash_normalization_and_truncation():
    from database import generate_desc_hash

    # Empty or default descriptions return empty string
    assert generate_desc_hash("") == ""
    assert generate_desc_hash("Description not available.") == ""

    # Normalization: punctuation, whitespace, and case
    desc1 = "Looking for an Embedded Firmware Engineer with C/C++ & RTOS experience!"
    desc2 = "  looking for an embedded firmware engineer with c c++   rtos experience   "
    h1 = generate_desc_hash(desc1)
    h2 = generate_desc_hash(desc2)
    assert len(h1) == 16
    assert h1 == h2

    # Different text yields different hash
    h3 = generate_desc_hash("Looking for a Python Django backend developer.")
    assert h1 != h3

    # First 500 characters determines hash; changes beyond 500 chars do not alter hash
    base_prefix = "a" * 500
    h_prefix1 = generate_desc_hash(base_prefix + " footer edit 12345")
    h_prefix2 = generate_desc_hash(base_prefix + " completely different company footer text")
    assert h_prefix1 == h_prefix2


def test_is_job_duplicate_and_desc_hash_query():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from database import is_job_duplicate, is_job_seen, generate_desc_hash

    mock_client = MagicMock()
    mock_client.execute_query = AsyncMock(
        return_value={
            "results": [
                {
                    "type": "ok",
                    "response": {
                        "type": "execute",
                        "result": {"rows": [[{"type": "integer", "value": 1}]]},
                    },
                }
            ]
        }
    )
    mock_client.session = None

    desc = "Unique firmware job description for medical devices with telemetry."
    expected_hash = generate_desc_hash(desc)

    # 1. is_job_seen checks desc_hash when description is passed
    seen = asyncio.run(is_job_seen(url_or_id="https://example.com/job1", title="Firmware", company="Stryker", description=desc, client=mock_client))
    assert seen is True
    query, args = mock_client.execute_query.call_args[0]
    assert "desc_hash = ?" in query
    assert expected_hash in args

    # 2. is_job_duplicate calls is_job_seen and catches duplicate
    dup = asyncio.run(is_job_duplicate(job_id="custom-id", description=desc, client=mock_client))
    assert dup is True


def test_schema_includes_retry_count_and_desc_hash():
    from database import SCHEMA, INDEXES
    assert "retry_count" in SCHEMA
    assert "desc_hash" in SCHEMA
    assert any("idx_jobs_retries" in idx for idx in INDEXES)


def test_async_database_writer_queue_processing():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from database import AsyncDatabaseWriter

    mock_client = MagicMock()
    mock_client.execute_query = AsyncMock(
        return_value={
            "results": [
                {
                    "type": "ok",
                    "response": {"type": "execute", "result": {"affected_row_count": 1}},
                }
            ]
        }
    )

    async def run_test():
        writer = AsyncDatabaseWriter(mock_client)
        writer.start()
        try:
            job_payload = {
                "job_id": "test-job-writer-1",
                "title": "Bio Engineer",
                "company": "Medtronic",
                "platform": "workday",
                "url": "https://example.com/job",
                "status": "active",
            }
            res = await writer.submit("add_job", job_payload)
            assert res is True
        finally:
            await writer.stop()

    asyncio.run(run_test())


def test_bulk_park_jobs():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from database import bulk_park_jobs

    mock_client = MagicMock()
    mock_client.execute_query = AsyncMock(return_value={"results": []})
    mock_client.session = None

    jobs = [
        {"title": "Job 1", "company": "Medtronic", "url": "https://m.com/1", "description": "Desc 1"},
        {"title": "Job 2", "company": "Stryker", "url": "https://s.com/2", "description": "Desc 2"},
    ]

    count = asyncio.run(bulk_park_jobs(jobs, reason="Test fallback limit", client=mock_client))
    assert count == 2
    query, args = mock_client.execute_query.call_args[0]
    assert "INSERT OR REPLACE INTO job_postings" in query
    assert "parked" in args


def test_increment_job_retry_discards_after_three_retries():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from database import increment_job_retry

    mock_client = MagicMock()
    mock_client.execute_query = AsyncMock(return_value={"results": []})
    mock_client.session = None

    res = asyncio.run(increment_job_retry("job-xyz-123", client=mock_client))
    assert res is True
    query, args = mock_client.execute_query.call_args[0]
    assert "retry_count = retry_count + 1" in query
    assert "CASE WHEN retry_count + 1 >= 3 THEN 'discarded'" in query
    assert args == ["job-xyz-123"]



