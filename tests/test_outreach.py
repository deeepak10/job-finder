"""Tests for Standalone Outreach Generator CLI Utility."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from outreach import generate_outreach_email, list_recent_jobs, search_jobs, copy_to_clipboard


def test_generate_outreach_email_success(monkeypatch):
    """Verify generate_outreach_email fetches job from Turso and formats Groq prompt."""
    fake_job = {
        "job_id": "test-job-999",
        "title": "Senior Firmware Engineer",
        "company": "Medtronic",
        "location": "Minneapolis, MN",
        "description": "Design safety-critical pacemaker firmware using C and RTOS.",
        "ai_score": 92,
        "job_category": "ECE_Hardware",
    }

    mock_client = MagicMock()
    mock_client.session = None

    async def mock_get_job(jid, client=None):
        return fake_job

    monkeypatch.setattr("outreach.get_job", mock_get_job)
    monkeypatch.setattr("outreach.get_turso_client", lambda: mock_client)
    monkeypatch.setattr("config.GROQ_API_KEY", "test-groq-key")

    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = (
        "SUBJECT: Senior Firmware Engineer - Medtronic Pacemaker Systems\n\n"
        "Hi Medtronic Team,\n\n"
        "I noticed your search for safety-critical pacemaker firmware engineers. "
        "With deep C/RTOS background, I would love to connect.\n\n"
        "Best regards,\nDeepak"
    )
    mock_completion.choices = [mock_choice]

    mock_groq_client = MagicMock()
    mock_groq_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    monkeypatch.setattr("outreach.AsyncOpenAI", lambda **kwargs: mock_groq_client)

    result = asyncio.run(generate_outreach_email("test-job-999"))
    assert result is not None
    assert "SUBJECT:" in result
    assert "Medtronic" in result
    assert "pacemaker" in result.lower()


def test_generate_outreach_email_job_not_found(monkeypatch, capsys):
    """Verify error message when job_id is missing in Turso."""
    mock_client = MagicMock()
    mock_client.session = None

    async def mock_get_job(jid, client=None):
        return None

    monkeypatch.setattr("outreach.get_job", mock_get_job)
    monkeypatch.setattr("outreach.get_turso_client", lambda: mock_client)

    result = asyncio.run(generate_outreach_email("non-existent-id"))
    assert result is None
    captured = capsys.readouterr()
    assert "not found in Turso database" in captured.out


def test_list_recent_jobs(monkeypatch, capsys):
    """Verify list_recent_jobs queries Turso and outputs table."""
    mock_client = MagicMock()
    mock_client.session = None
    mock_client.execute_query = AsyncMock(return_value={
        "results": [
            {
                "type": "ok",
                "response": {
                    "type": "execute",
                    "result": {
                        "cols": [{"name": "job_id"}, {"name": "title"}, {"name": "company"}, {"name": "ai_score"}, {"name": "job_category"}, {"name": "date_found"}],
                        "rows": [
                            [
                                {"type": "text", "value": "job-101"},
                                {"type": "text", "value": "Biomedical Systems Engineer"},
                                {"type": "text", "value": "Stryker"},
                                {"type": "integer", "value": 88},
                                {"type": "text", "value": "Biomedical_RD"},
                                {"type": "text", "value": "2026-09-22T10:00:00Z"},
                            ]
                        ],
                    },
                },
            }
        ]
    })
    monkeypatch.setattr("outreach.get_turso_client", lambda: mock_client)

    asyncio.run(list_recent_jobs(limit=5))
    captured = capsys.readouterr()
    assert "RECENT HIGH-MATCH JOBS" in captured.out
    assert "job-101" in captured.out
    assert "Stryker" in captured.out


def test_copy_to_clipboard():
    """Verify copy_to_clipboard runs safely without raising uncaught exceptions."""
    # Should either succeed or return boolean False gracefully
    res = copy_to_clipboard("Test message content")
    assert isinstance(res, bool)
