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
