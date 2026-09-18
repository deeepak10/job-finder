"""Tests for Turso Cloud Database layer."""

import pytest
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
    }

    success = asyncio.run(add_job(job_dict, client=mock_client))
    assert success is True
    mock_client.execute_query.assert_called_once()
    call_args = mock_client.execute_query.call_args[0]
    sql, args = call_args[0], call_args[1]
    assert "ai_reasoning" not in sql
    assert "portfolio_highlight" not in sql
    assert "outreach_message" not in sql
    assert len(args) == 10  # job_id, title, company, location, platform, url, ai_score, visa_sponsorship, date_found, alert_sent
    assert "Some reasoning" not in args
    assert "Project A" not in args
    assert "Draft" not in args
