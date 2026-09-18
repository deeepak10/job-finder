"""Tests for quota_manager.py"""

import pytest
from unittest.mock import patch, MagicMock

import quota_manager


def test_get_current_section():
    section = quota_manager.get_current_section()
    assert section in (0, 1, 2)


@patch("quota_manager.get_serpapi_account_info")
def test_budget_calculation_handles_normal_quota(mock_info):
    mock_info.return_value = {
        "searches_left": 240,
        "total_limit": 250,
        "renewal_date": "2026-10-15",
    }
    budget, locs = quota_manager.calculate_section_search_budget()
    assert budget > 0
    assert len(locs) == budget
    assert len(locs) <= 3


@patch("quota_manager.get_serpapi_account_info")
def test_budget_pauses_when_quota_critical(mock_info):
    mock_info.return_value = {
        "searches_left": 3,
        "total_limit": 250,
        "renewal_date": "2026-10-15",
    }
    budget, locs = quota_manager.calculate_section_search_budget()
    assert budget == 0
    assert locs == []


def test_get_active_locations_chunks_into_three(monkeypatch):
    """Verify get_active_locations splits locations into safe chunks."""
    import config
    # 14 locations in SERPAPI_LOCATIONS
    assert len(config.SERPAPI_LOCATIONS) == 14

    locs = quota_manager.get_active_locations()
    assert isinstance(locs, list)
    assert len(locs) > 0
    # With 14 items chunked into 3, chunks are size 5, 5, or 4
    assert len(locs) in (4, 5)
    # Every item in chunk should be a valid string in SERPAPI_LOCATIONS
    for loc in locs:
        assert loc in config.SERPAPI_LOCATIONS


def test_get_active_locations_rotation(monkeypatch):
    """Verify rotation returns different chunks based on hour."""
    from datetime import datetime, timezone

    class MockDateTime(datetime):
        _hour = 0
        @classmethod
        def utcnow(cls):
            return cls(2026, 9, 18, cls._hour, 0, 0)
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 18, cls._hour, 0, 0, tzinfo=tz or timezone.utc)

    monkeypatch.setattr("quota_manager.datetime", MockDateTime)

    MockDateTime._hour = 0  # 0 % 3 = 0
    chunk_0 = quota_manager.get_active_locations()

    MockDateTime._hour = 1  # 1 % 3 = 1
    chunk_1 = quota_manager.get_active_locations()

    MockDateTime._hour = 2  # 2 % 3 = 2
    chunk_2 = quota_manager.get_active_locations()

    assert chunk_0 != chunk_1
    assert chunk_1 != chunk_2
    # Together all 3 chunks should cover all 14 locations
    combined = chunk_0 + chunk_1 + chunk_2
    assert len(combined) == 14
    import config
    assert combined == config.SERPAPI_LOCATIONS


def test_get_active_locations_empty_fallback(monkeypatch):
    """Verify get_active_locations falls back to ['India'] if empty."""
    import config
    monkeypatch.setattr(config, "SERPAPI_LOCATIONS", [])
    locs = quota_manager.get_active_locations()
    assert locs == ["India"]

