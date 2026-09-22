"""Tests for Layer 1 Groq Coarse Batch Pre-Filtering."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from evaluator import batch_coarse_filter_groq
from scrapers.base import JobResult


def test_batch_coarse_filter_empty_list():
    relevant, discarded = asyncio.run(batch_coarse_filter_groq([]))
    assert relevant == []
    assert discarded == []


def test_batch_coarse_filter_prunes_non_technical_roles(monkeypatch):
    """Verify Groq batch pre-filter separates technical roles from non-technical ones."""
    jobs = [
        JobResult(title="Senior Biomedical Systems Engineer", company="Medtronic", url="https://m.com/1", platform="workday"),
        JobResult(title="Registered Nurse - ICU Ward", company="Apollo Hospitals", url="https://a.com/2", platform="naukri"),
        JobResult(title="Principal Firmware Engineer (C/C++)", company="Stryker", url="https://s.com/3", platform="workday"),
        JobResult(title="Medical Billing & Insurance Specialist", company="HealthCare Corp", url="https://h.com/4", platform="adzuna"),
        JobResult(title="Python Full Stack HealthTech Developer", company="GE HealthCare", url="https://g.com/5", platform="linkedin"),
    ]

    mock_client = MagicMock()
    mock_choice = MagicMock()
    # Mock Groq returning relevant_ids for indices 0, 2, 4 (technical roles)
    mock_choice.message.content = json.dumps({"relevant_ids": [0, 2, 4]})
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    monkeypatch.setattr("config.GROQ_API_KEY", "test-key")

    relevant, discarded = asyncio.run(batch_coarse_filter_groq(jobs, groq_cli=mock_client))

    assert len(relevant) == 3
    assert len(discarded) == 2

    relevant_titles = [j.title for j in relevant]
    discarded_titles = [j.title for j in discarded]

    assert "Senior Biomedical Systems Engineer" in relevant_titles
    assert "Principal Firmware Engineer (C/C++)" in relevant_titles
    assert "Python Full Stack HealthTech Developer" in relevant_titles

    assert "Registered Nurse - ICU Ward" in discarded_titles
    assert "Medical Billing & Insurance Specialist" in discarded_titles


def test_batch_coarse_filter_fails_open_on_groq_error(monkeypatch):
    """Ensure filter fails open (returns all jobs) if Groq returns an error or malformed JSON."""
    jobs = [
        JobResult(title="Biomedical Engineer", company="Medtronic", url="https://m.com/1", platform="workday"),
        JobResult(title="Firmware Developer", company="Philips", url="https://p.com/2", platform="workday"),
    ]

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("Groq 503 Service Unavailable"))

    monkeypatch.setattr("config.GROQ_API_KEY", "test-key")

    relevant, discarded = asyncio.run(batch_coarse_filter_groq(jobs, groq_cli=mock_client))

    # Must fail open: preserve all jobs without dropping them
    assert len(relevant) == 2
    assert len(discarded) == 0


def test_batch_coarse_filter_bypasses_when_no_api_key(monkeypatch):
    """Ensure filter cleanly bypasses when GROQ_API_KEY is unset or mock-key."""
    jobs = [
        JobResult(title="Biomedical Engineer", company="Medtronic", url="https://m.com/1", platform="workday"),
    ]
    monkeypatch.setattr("config.GROQ_API_KEY", "")

    relevant, discarded = asyncio.run(batch_coarse_filter_groq(jobs))
    assert len(relevant) == 1
    assert len(discarded) == 0
