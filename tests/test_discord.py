"""Tests for Discord Notification Engine (v2)."""

import pytest
from unittest.mock import AsyncMock, patch

from discord_alerts import (
    COLOR_BLUE,
    COLOR_PURPLE,
    build_discord_embed,
    pick_embed_color,
    send_discord_alert_async,
)
from evaluator import JobEvaluation


@pytest.fixture
def sample_evaluation():
    return JobEvaluation(
        is_match=True,
        visa_sponsorship="Work authorization supported",
    )


@pytest.fixture
def sample_job():
    return {
        "title": "Biomedical Systems Engineer",
        "company": "Philips Healthcare",
        "location": "Bengaluru, India",
        "platform": "naukri",
        "url": "https://naukri.com/job/123",
    }


def test_pick_embed_color_target_companies():
    assert pick_embed_color("Philips Healthcare") == COLOR_PURPLE
    assert pick_embed_color("GE HealthCare") == COLOR_PURPLE
    assert pick_embed_color("Stryker Global") == COLOR_PURPLE
    assert pick_embed_color("Dozee Health") == COLOR_PURPLE
    assert pick_embed_color("Qure.ai Technologies") == COLOR_PURPLE
    assert pick_embed_color("Skanray Technologies") == COLOR_PURPLE
    assert pick_embed_color("Agappe Diagnostics Ltd") == COLOR_PURPLE


def test_pick_embed_color_standard_companies():
    assert pick_embed_color("Random Startup Inc") == COLOR_BLUE
    assert pick_embed_color("Acme Software Labs") == COLOR_BLUE


def test_embed_structure(sample_job, sample_evaluation):
    embed = build_discord_embed(sample_job, sample_evaluation)

    # Clickable title with green box emoji
    assert embed["title"] == "🟩 Biomedical Systems Engineer"
    assert embed["url"] == "https://naukri.com/job/123"
    assert embed["color"] == COLOR_PURPLE

    # Inline and block fields
    field_dict = {f["name"]: f for f in embed["fields"]}
    assert "🏢 Company" in field_dict and field_dict["🏢 Company"]["inline"] is True
    assert "📍 Location" in field_dict and field_dict["📍 Location"]["inline"] is True
    assert "🌐 Platform" in field_dict and field_dict["🌐 Platform"]["inline"] is True
    assert "🛂 Visa Status" in field_dict and field_dict["🛂 Visa Status"]["inline"] is True
    assert "🧠 AI Reasoning" not in field_dict

    # Removed fields to eliminate clutter and token consumption
    assert "🔥 AI Match Score" not in field_dict
    assert "✉️ Outreach Draft" not in field_dict
    assert "footer" not in embed or "💡 Tip" not in embed.get("footer", {}).get("text", "")


def test_alert_gating_is_match_false(sample_job, sample_evaluation):
    import asyncio
    sample_evaluation.is_match = False
    with patch("aiohttp.ClientSession.post") as mock_post:
        sent = asyncio.run(send_discord_alert_async(sample_job, sample_evaluation))
        assert sent is False
        mock_post.assert_not_called()


def test_alert_gating_is_match_true(sample_job, sample_evaluation):
    import asyncio
    sample_evaluation.is_match = True
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 204
        sent = asyncio.run(send_discord_alert_async(sample_job, sample_evaluation, webhook_url="https://discord.com/api/webhooks/123/abc"))
        assert sent is True
        mock_post.assert_called_once()

