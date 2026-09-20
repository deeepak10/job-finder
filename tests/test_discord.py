"""Tests for Discord Notification Engine (v2)."""

import pytest
from unittest.mock import patch

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
    # Context field for why role was matched
    assert "Why it matched:" in field_dict and field_dict["Why it matched:"]["inline"] is False
    assert "🏷️ Domain" in field_dict and field_dict["🏷️ Domain"]["inline"] is True
    assert embed.get("footer", {}).get("text") == "Tri-Model AI Routing Pipeline"

    # Removed fields to eliminate clutter and token consumption
    assert "🔥 AI Match Score" not in field_dict
    assert "✉️ Outreach Draft" not in field_dict
    assert "💡 Tip" not in embed.get("footer", {}).get("text", "")


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


def test_main_imports_send_discord_alert_async():
    import main
    assert hasattr(main, "send_discord_alert_async")
    assert callable(main.send_discord_alert_async)
    assert hasattr(main, "send_system_alert_async")
    assert callable(main.send_system_alert_async)


def test_send_system_alert_async():
    import asyncio
    from discord_alerts import send_system_alert_async

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 204
        sent = asyncio.run(send_system_alert_async("Test failure alert", webhook_url="https://discord.com/api/webhooks/123/abc"))
        assert sent is True
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        embed = call_kwargs["json"]["embeds"][0]
        assert embed["color"] == 16711680
        assert "Pipeline Anomaly Detected" in embed["title"]
        assert "Test failure alert" in embed["description"]
        assert embed["footer"]["text"] == "Job Finder System Monitor"


def test_multi_channel_discord_routing(sample_job, monkeypatch):
    import asyncio
    import config

    monkeypatch.setattr(config, "DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/default/111")
    monkeypatch.setattr(config, "DISCORD_WEBHOOK_DEFAULT", "https://discord.com/api/webhooks/default/111")
    monkeypatch.setattr(config, "DISCORD_WEBHOOK_BIOMED", "https://discord.com/api/webhooks/biomed/222")
    monkeypatch.setattr(config, "DISCORD_WEBHOOK_ECE", "https://discord.com/api/webhooks/ece/333")
    monkeypatch.setattr(config, "DISCORD_WEBHOOK_SOFTWARE", "https://discord.com/api/webhooks/software/444")

    # 1. Route Biomedical_RD
    bio_eval = JobEvaluation(is_match=True, visa_sponsorship="Supported", job_category="Biomedical_RD")
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 200
        res = asyncio.run(send_discord_alert_async(sample_job, bio_eval))
        assert res is True
        assert mock_post.call_args[0][0] == "https://discord.com/api/webhooks/biomed/222"

    # 2. Route ECE_Hardware
    ece_eval = JobEvaluation(is_match=True, visa_sponsorship="Supported", job_category="ECE_Hardware")
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 200
        res = asyncio.run(send_discord_alert_async(sample_job, ece_eval))
        assert res is True
        assert mock_post.call_args[0][0] == "https://discord.com/api/webhooks/ece/333"

    # 3. Route Software_Web
    sw_eval = JobEvaluation(is_match=True, visa_sponsorship="Supported", job_category="Software_Web")
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 200
        res = asyncio.run(send_discord_alert_async(sample_job, sw_eval))
        assert res is True
        assert mock_post.call_args[0][0] == "https://discord.com/api/webhooks/software/444"

    # 4. Fallback to default when category is General
    gen_eval = JobEvaluation(is_match=True, visa_sponsorship="Supported", job_category="General")
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 200
        res = asyncio.run(send_discord_alert_async(sample_job, gen_eval))
        assert res is True
        assert mock_post.call_args[0][0] == "https://discord.com/api/webhooks/default/111"

    # 5. Fallback to default when category webhook is unset in config
    monkeypatch.setattr(config, "DISCORD_WEBHOOK_BIOMED", "")
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 200
        res = asyncio.run(send_discord_alert_async(sample_job, bio_eval))
        assert res is True
        assert mock_post.call_args[0][0] == "https://discord.com/api/webhooks/default/111"


def test_multi_channel_discord_routing_safe_type_handling(sample_job, monkeypatch):
    import asyncio
    import config

    monkeypatch.setattr(config, "DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/default/111")
    monkeypatch.setattr(config, "DISCORD_WEBHOOK_DEFAULT", "https://discord.com/api/webhooks/default/111")

    # 1. When evaluation is a dict with job_category = None
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 200
        res = asyncio.run(send_discord_alert_async(sample_job, {"is_match": True, "job_category": None}))
        assert res is True
        assert mock_post.call_args[0][0] == "https://discord.com/api/webhooks/default/111"

    # 2. When evaluation has an unknown/unregistered category string
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 200
        res = asyncio.run(send_discord_alert_async(sample_job, {"is_match": True, "job_category": "Quantum_Computing"}))
        assert res is True
        assert mock_post.call_args[0][0] == "https://discord.com/api/webhooks/default/111"

    # 3. When evaluation has a non-string category (e.g. integer)
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.status = 200
        res = asyncio.run(send_discord_alert_async(sample_job, {"is_match": True, "job_category": 12345}))
        assert res is True
        assert mock_post.call_args[0][0] == "https://discord.com/api/webhooks/default/111"



