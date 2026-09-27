import asyncio
from unittest.mock import AsyncMock, MagicMock

import config
from evaluator import (
    JobEvaluation,
    evaluate_job_consensus,
    evaluate_with_consensus,
    query_model,
)


def test_query_model_success():
    """Ensure query_model parses structured JSON correctly."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"is_match": true, "visa_sponsorship": "Yes", "ai_score": 88}'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    res = asyncio.run(query_model(mock_client, "test-model", "Test prompt"))
    assert res == {"is_match": True, "visa_sponsorship": "Yes", "ai_score": 88}


def test_query_model_strips_code_fences():
    """Ensure query_model handles markdown json code fences."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '```json\n{"is_match": false, "visa_sponsorship": "No", "ai_score": 20}\n```'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    res = asyncio.run(query_model(mock_client, "test-model", "Test prompt"))
    assert res == {"is_match": False, "visa_sponsorship": "No", "ai_score": 20}


def test_query_model_handles_exception():
    """Ensure query_model returns None safely on failure."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API connection refused"))

    res = asyncio.run(query_model(mock_client, "test-model", "Test prompt"))
    assert res is None


def test_evaluate_with_consensus_agreement(monkeypatch):
    """Ensure evaluate_with_consensus returns consensus_passed when both models match."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "mock-groq-key")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "mock-router-key")

    mock_groq = MagicMock()
    mock_groq_choice = MagicMock()
    mock_groq_choice.message.content = '{"is_match": true, "visa_sponsorship": "Supported (US H1B)", "ai_score": 90}'
    mock_groq_res = MagicMock(choices=[mock_groq_choice])
    mock_groq.chat.completions.create = AsyncMock(return_value=mock_groq_res)

    mock_router = MagicMock()
    mock_router_choice = MagicMock()
    mock_router_choice.message.content = '{"is_match": true, "visa_sponsorship": "Supported (US H1B)", "ai_score": 80}'
    mock_router_res = MagicMock(choices=[mock_router_choice])
    mock_router.chat.completions.create = AsyncMock(return_value=mock_router_res)

    result = asyncio.run(evaluate_with_consensus("Prompt", groq_cli=mock_groq, router_cli=mock_router))

    assert result["is_match"] is True
    assert result["status"] == "consensus_passed"
    assert result["ai_score"] == 85
    assert result["visa_sponsorship"] == "Supported (US H1B)"


def test_evaluate_with_consensus_conflict(monkeypatch):
    """Ensure evaluate_with_consensus returns conflict when models disagree."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "mock-groq-key")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "mock-router-key")

    mock_groq = MagicMock()
    mock_groq_choice = MagicMock()
    mock_groq_choice.message.content = '{"is_match": true, "visa_sponsorship": "Supported", "ai_score": 85}'
    mock_groq.chat.completions.create = AsyncMock(return_value=MagicMock(choices=[mock_groq_choice]))

    mock_router = MagicMock()
    mock_router_choice = MagicMock()
    mock_router_choice.message.content = '{"is_match": false, "visa_sponsorship": "Unknown", "ai_score": 35}'
    mock_router.chat.completions.create = AsyncMock(return_value=MagicMock(choices=[mock_router_choice]))

    result = asyncio.run(evaluate_with_consensus("Prompt", groq_cli=mock_groq, router_cli=mock_router))

    assert result["is_match"] is False
    assert result["status"] == "conflict"
    assert result["ai_score"] == 0


def test_evaluate_with_consensus_both_rejected(monkeypatch):
    """Ensure evaluate_with_consensus returns consensus_failed when both models reject."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "mock-groq-key")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "mock-router-key")

    mock_groq = MagicMock()
    mock_groq_choice = MagicMock()
    mock_groq_choice.message.content = '{"is_match": false, "visa_sponsorship": "None", "ai_score": 20}'
    mock_groq.chat.completions.create = AsyncMock(return_value=MagicMock(choices=[mock_groq_choice]))

    mock_router = MagicMock()
    mock_router_choice = MagicMock()
    mock_router_choice.message.content = '{"is_match": false, "visa_sponsorship": "None", "ai_score": 25}'
    mock_router.chat.completions.create = AsyncMock(return_value=MagicMock(choices=[mock_router_choice]))

    result = asyncio.run(evaluate_with_consensus("Prompt", groq_cli=mock_groq, router_cli=mock_router))

    assert result["is_match"] is False
    assert result["status"] == "consensus_failed"
    assert result["ai_score"] == 0


def test_evaluate_with_consensus_failure(monkeypatch):
    """Ensure evaluate_with_consensus defers job when a model fails."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "mock-groq-key")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "mock-router-key")

    mock_groq = MagicMock()
    mock_groq.chat.completions.create = AsyncMock(side_effect=Exception("503 Service Unavailable"))

    mock_router = MagicMock()
    mock_router_choice = MagicMock()
    mock_router_choice.message.content = '{"is_match": true, "ai_score": 80}'
    mock_router.chat.completions.create = AsyncMock(return_value=MagicMock(choices=[mock_router_choice]))

    result = asyncio.run(evaluate_with_consensus("Prompt", groq_cli=mock_groq, router_cli=mock_router))

    assert result["is_match"] is False
    assert result["status"] == "deferred"


def test_evaluate_with_consensus_missing_keys(monkeypatch):
    """Ensure evaluate_with_consensus defers cleanly when API keys are missing."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")

    result = asyncio.run(evaluate_with_consensus("Prompt"))
    assert result["is_match"] is False
    assert result["status"] == "deferred"


def test_evaluate_job_consensus_wrapper(monkeypatch):
    """Ensure evaluate_job_consensus constructs a JobEvaluation with status."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "mock-key")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "mock-key")

    mock_groq = MagicMock()
    mock_groq.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content='{"is_match": true, "visa_sponsorship": "Direct", "ai_score": 92}'))])
    )
    mock_router = MagicMock()
    mock_router.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content='{"is_match": true, "visa_sponsorship": "Direct", "ai_score": 88}'))])
    )

    job_dict = {
        "title": "Biomedical Firmware Engineer",
        "company": "Medtronic",
        "location": "Bengaluru",
        "platform": "Workday ATS",
        "description": "Embedded C for implantable telemetry",
    }

    eval_result = asyncio.run(evaluate_job_consensus(job_dict, groq_cli=mock_groq, router_cli=mock_router))

    assert isinstance(eval_result, JobEvaluation)
    assert eval_result.is_match is True
    assert eval_result.status == "consensus_passed"
    assert eval_result.ai_score == 90
    assert eval_result.match_score == 90


def test_database_deferred_jobs_helpers():
    """Verify get_deferred_jobs and update_job_status against mock Turso client."""
    from database import get_deferred_jobs, update_job_status

    mock_client = MagicMock()
    mock_client.session = None

    # Test get_deferred_jobs
    mock_rows = [
        {
            "job_id": "test1234",
            "title": "Firmware Engineer",
            "company": "Philips",
            "status": "deferred",
            "description": "Full description text",
        }
    ]
    mock_client.execute_query = AsyncMock(
        return_value={
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
        }
    )

    jobs = asyncio.run(get_deferred_jobs(limit=10, client=mock_client))
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == "test1234"
    assert jobs[0]["status"] == "deferred"

    # Test update_job_status
    mock_client.execute_query = AsyncMock(return_value={"rows_affected": 1})
    updated = asyncio.run(
        update_job_status(
            job_id="test1234",
            status="consensus_passed",
            ai_score=85,
            visa_sponsorship="Supported",
            alert_sent=1,
            client=mock_client,
        )
    )
    assert updated is True
