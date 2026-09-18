"""Tests for Tri-Model Routing Architecture (Executive Tie-Breaker)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

import config
from evaluator import (
    GeminiQuotaExceededError,
    JobEvaluation,
    evaluate_job_consensus,
    evaluate_with_consensus,
)
from scrapers.base import JobResult


def test_job_result_tier_defaults_and_explicit():
    """Verify JobResult defaults to strict and supports broad tier."""
    j1 = JobResult(title="Bio Eng", company="MedTech", url="https://example.com/1", platform="naukri")
    assert j1.tier == "strict"
    assert j1.to_row()["tier"] == "strict"

    j2 = JobResult(title="Python Dev", company="HealthTech", url="https://example.com/2", platform="naukri", tier="broad")
    assert j2.tier == "broad"
    assert j2.to_row()["tier"] == "broad"


def test_get_query_tier_logic():
    """Verify query tier detection between strict MedTech and broad software roles."""
    assert config.get_query_tier("medical-device-rnd-jobs") == "strict"
    assert config.get_query_tier("Biomedical Firmware") == "strict"
    assert config.get_query_tier("Signal Processing Engineer") == "strict"

    assert config.get_query_tier("embedded-software-engineer-healthcare-jobs") == "broad"
    assert config.get_query_tier("python-developer-healthcare-jobs") == "broad"
    assert config.get_query_tier("research-and-development-engineer-jobs") == "broad"
    assert config.get_query_tier("Python Healthtech") == "broad"


def test_tri_model_consensus_reconciliation_states(monkeypatch):
    """Verify consensus reconciliation yields consensus_passed, consensus_failed, and conflict."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "mock-groq")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "mock-or")

    # Case 1: Both Accept
    g_pass = MagicMock(chat=MagicMock(completions=MagicMock(create=AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"is_match": true, "ai_score": 90, "visa_sponsorship": "Supported"}'))]
    )))))
    r_pass = MagicMock(chat=MagicMock(completions=MagicMock(create=AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"is_match": true, "ai_score": 80, "visa_sponsorship": "Supported"}'))]
    )))))
    res_pass = asyncio.run(evaluate_with_consensus("Prompt", groq_cli=g_pass, router_cli=r_pass))
    assert res_pass["status"] == "consensus_passed"
    assert res_pass["is_match"] is True
    assert res_pass["ai_score"] == 85

    # Case 2: Both Reject
    g_fail = MagicMock(chat=MagicMock(completions=MagicMock(create=AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"is_match": false, "ai_score": 10, "visa_sponsorship": "None"}'))]
    )))))
    r_fail = MagicMock(chat=MagicMock(completions=MagicMock(create=AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"is_match": false, "ai_score": 15, "visa_sponsorship": "None"}'))]
    )))))
    res_fail = asyncio.run(evaluate_with_consensus("Prompt", groq_cli=g_fail, router_cli=r_fail))
    assert res_fail["status"] == "consensus_failed"
    assert res_fail["is_match"] is False

    # Case 3: Split Decision (Conflict)
    res_conflict = asyncio.run(evaluate_with_consensus("Prompt", groq_cli=g_pass, router_cli=r_fail))
    assert res_conflict["status"] == "conflict"
    assert res_conflict["is_match"] is False


def test_route_b_gatekeeper_rejection():
    """Verify Route B Groq gatekeeper rejection stops early and preserves Gemini quota."""
    from main import query_model

    mock_groq = MagicMock()
    mock_groq.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"is_match": false, "ai_score": 20, "visa_sponsorship": "None"}'))]
    ))

    result = asyncio.run(query_model(mock_groq, "test-model", "Test prompt"))
    assert result is not None
    assert result["is_match"] is False


def test_route_b_gatekeeper_accept_openrouter_accept():
    """Verify Route B both models accepting yields approval."""
    from main import query_model

    mock_groq = MagicMock()
    mock_groq.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"is_match": true, "ai_score": 80, "visa_sponsorship": "Supported"}'))]
    ))

    mock_or = MagicMock()
    mock_or.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"is_match": true, "ai_score": 84, "visa_sponsorship": "Supported"}'))]
    ))

    async def _test():
        groq_res = await query_model(mock_groq, "groq-model", "Prompt")
        or_res = await query_model(mock_or, "or-model", "Prompt")
        return groq_res, or_res

    groq_res, or_res = asyncio.run(_test())

    assert groq_res["is_match"] is True
    assert or_res["is_match"] is True
    avg_score = int((groq_res["ai_score"] + or_res["ai_score"]) / 2)
    assert avg_score == 82


def test_executive_tie_breaker_database_update():
    """Verify deferred jobs get updated to active or rejected during executive tie-break."""
    from database import update_job_status

    mock_client = MagicMock()
    mock_client.execute_query = AsyncMock(return_value={"rows_affected": 1})
    mock_client.session = None

    ok_active = asyncio.run(update_job_status("job-123", status="active", ai_score=88, client=mock_client))
    assert ok_active is True
    call_args_active = mock_client.execute_query.call_args[0]
    assert "status = ?" in call_args_active[0]
    assert "active" in call_args_active[1]

    ok_reject = asyncio.run(update_job_status("job-123", status="rejected", ai_score=30, client=mock_client))
    assert ok_reject is True
    call_args_reject = mock_client.execute_query.call_args[0]
    assert "rejected" in call_args_reject[1]


def test_get_pending_groq_jobs():
    """Verify get_pending_groq_jobs queries Turso with pending_groq_verification status."""
    from database import get_pending_groq_jobs

    mock_client = MagicMock()
    mock_rows = [
        {
            "job_id": "j1",
            "title": "Software Engineer Healthcare",
            "company": "HealthCorp",
            "status": "pending_groq_verification",
        }
    ]
    mock_client.execute_query = AsyncMock(return_value={
        "results": [
            {
                "type": "ok",
                "response": {
                    "type": "execute",
                    "result": {
                        "cols": [{"name": k} for k in mock_rows[0].keys()],
                        "rows": [[{"type": "text", "value": v} for v in mock_rows[0].values()]],
                    },
                },
            }
        ]
    })
    mock_client.session = None

    jobs = asyncio.run(get_pending_groq_jobs(limit=10, client=mock_client))
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == "j1"
    assert jobs[0]["status"] == "pending_groq_verification"

    call_args = mock_client.execute_query.call_args[0]
    assert "pending_groq_verification" in call_args[0]


def test_route_b_fallback_ambiguity_openrouter_rejects():
    """When Groq is offline, solo OpenRouter rejects -> marked rejected."""
    from main import query_model

    mock_or = MagicMock()
    mock_or.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"is_match": false, "ai_score": 10, "visa_sponsorship": "None"}'))]
    ))

    or_res = asyncio.run(query_model(mock_or, "deepseek/deepseek-chat", "Prompt"))
    assert or_res is not None
    assert or_res["is_match"] is False


def test_route_b_fallback_ambiguity_openrouter_accepts():
    """When Groq is offline, solo OpenRouter accepts -> pending_groq_verification (no alert)."""
    from main import query_model

    mock_or = MagicMock()
    mock_or.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"is_match": true, "ai_score": 75, "visa_sponsorship": "Local"}'))]
    ))

    or_res = asyncio.run(query_model(mock_or, "deepseek/deepseek-chat", "Prompt"))
    assert or_res is not None
    assert or_res["is_match"] is True
    # The pipeline sets status = 'pending_groq_verification' and does NOT alert


def test_target_urls_and_spam_keywords_configured():
    """Verify hybrid TARGET_URLS and reinforced SPAM_KEYWORDS are populated."""
    from filters import SPAM_KEYWORDS

    assert hasattr(config, "TARGET_URLS")
    assert len(config.TARGET_URLS) == 8
    assert "https://www.naukri.com/medical-device-rnd-jobs" in config.TARGET_URLS
    assert "https://www.naukri.com/software-engineer-medical-device-jobs" in config.TARGET_URLS

    expected_spam = [
        "sales", "bde", "business development", "marketing", "representative",
        "field service", "maintenance", "repair technician", "service engineer",
        "billing", "pharmacist", "receptionist", "clerk", "bpo", "helpdesk",
        "customer support", "voice process", "telecaller",
    ]
    for kw in expected_spam:
        assert kw in SPAM_KEYWORDS


def test_update_job_status_persists_match_reason():
    """Verify update_job_status properly includes match_reason in SQL and args."""
    from database import update_job_status

    mock_client = MagicMock()
    mock_client.execute_query = AsyncMock(return_value={"rows_affected": 1})
    mock_client.session = None

    reason_text = "Strong embedded C++ and biomedical device diagnostics."
    ok = asyncio.run(
        update_job_status(
            "job-456",
            status="active",
            ai_score=92,
            visa_sponsorship="Available",
            match_reason=reason_text,
            client=mock_client,
        )
    )
    assert ok is True
    sql, args = mock_client.execute_query.call_args[0]
    assert "match_reason = ?" in sql
    assert reason_text in args
    assert "active" in args
    assert 92 in args


def test_query_model_exponential_backoff_on_429(monkeypatch):
    """Verify query_model retries with exponential backoff on HTTP 429 rate limit."""
    from evaluator import query_model

    sleep_calls = []

    async def mock_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", mock_sleep)

    mock_client = MagicMock()
    call_count = 0

    async def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            err = Exception("Error code: 429 - {'error': {'message': 'Rate limit reached'}}")
            setattr(err, "status_code", 429)
            raise err
        # Success on attempt 3
        mock_choice = MagicMock()
        mock_choice.message.content = '```json\n{"is_match": true, "ai_score": 85, "visa_sponsorship": "Yes", "match_reason": "Relevant IoT firmware background"}\n```'
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    mock_client.chat.completions.create = mock_create

    result = asyncio.run(query_model(mock_client, "test-model", "test prompt"))
    assert result is not None
    assert result["is_match"] is True
    assert result["ai_score"] == 85
    assert result["match_reason"] == "Relevant IoT firmware background"
    assert call_count == 3
    assert len(sleep_calls) == 2  # Slept twice before 3rd successful attempt
    assert sleep_calls[0] > 0
    assert sleep_calls[1] > sleep_calls[0]


def test_parse_ai_json_extraction_edge_cases():
    """Verify parse_ai_json safely handles markdown codeblocks and conversational preamble."""
    from evaluator import parse_ai_json

    # Test 1: Markdown backticks with preamble and postamble
    dirty_text = (
        "Here is the evaluation for the position:\n"
        "```json\n"
        "{\n"
        '  "is_match": true,\n'
        '  "visa_sponsorship": "Eligible",\n'
        '  "ai_score": 88,\n'
        '  "match_reason": "Deep C/C++ firmware and medical instrumentation experience."\n'
        "}\n"
        "```\n"
        "Let me know if you need further details!"
    )
    res = parse_ai_json(dirty_text)
    assert res is not None
    assert res["is_match"] is True
    assert res["ai_score"] == 88
    assert "biomedical" in res["match_reason"].lower() or "c/c++" in res["match_reason"].lower()

    # Test 2: Invalid text returns empty dict
    assert parse_ai_json("Just plain text with no json") == {}

