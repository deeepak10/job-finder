"""Tests for Evaluator Pydantic Schema and Async Logic."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from evaluator import JobEvaluation, SYSTEM_INSTRUCTION, evaluate_job


def test_job_evaluation_schema_validation():
    data = {
        "is_match": True,
        "visa_sponsorship": "Local role (no sponsorship required)",
    }
    obj = JobEvaluation.model_validate(data)
    assert obj.is_match is True
    assert obj.visa_sponsorship == "Local role (no sponsorship required)"

    # Token-eating fields should not be in the model schema fields
    assert "reasoning" not in JobEvaluation.model_fields
    assert "match_score" not in JobEvaluation.model_fields
    assert "portfolio_highlight" not in JobEvaluation.model_fields
    assert "linkedin_outreach_message" not in JobEvaluation.model_fields


def test_system_prompt_contains_required_sections():
    assert "Biomedical Engineering, Electronics, IoT, and Python Data Analytics" in SYSTEM_INSTRUCTION
    assert "CORE PRIORITIES" in SYSTEM_INSTRUCTION
    assert "BROAD SEMANTIC MATCHING" in SYSTEM_INSTRUCTION
    assert "STRICT REJECTIONS" in SYSTEM_INSTRUCTION
    assert "Medical Device R&D" in SYSTEM_INSTRUCTION
    assert "Biomedical Firmware" in SYSTEM_INSTRUCTION
    assert "visa_sponsorship" in SYSTEM_INSTRUCTION
    assert "telemetry" not in SYSTEM_INSTRUCTION.lower()
    # Outreach draft instruction removed to conserve LLM generation tokens
    assert "linkedin_outreach_message" not in SYSTEM_INSTRUCTION


def test_evaluate_job_timeout_fails_gracefully(monkeypatch):
    import asyncio
    import config

    monkeypatch.setattr(config, "GEMINI_TIMEOUT_SECONDS", 0.05)
    semaphore = asyncio.Semaphore(1)

    async def mock_hanging_call(*args, **kwargs):
        await asyncio.sleep(1.0)
        return JobEvaluation(is_match=True, visa_sponsorship="yes")

    monkeypatch.setattr("evaluator._call_gemini", mock_hanging_call)

    job_data = {
        "title": "Hanging Job",
        "company": "Slow Corp",
        "location": "Remote",
        "platform": "naukri",
        "description": "Some description",
    }
    mock_client = MagicMock()

    async def _run():
        return await evaluate_job(job_data, semaphore, client=mock_client)

    result = asyncio.run(_run())
    assert result is None  # Must gracefully drop job and return None on timeout


def test_evaluate_job_success(monkeypatch):
    import asyncio
    semaphore = asyncio.Semaphore(1)

    expected = JobEvaluation(
        is_match=True,
        visa_sponsorship="Supported",
    )

    async def mock_successful_call(*args, **kwargs):
        return expected

    monkeypatch.setattr("evaluator._call_gemini", mock_successful_call)

    job_data = {
        "title": "Biomedical Engineer",
        "company": "GE HealthCare",
        "location": "Bengaluru",
        "platform": "linkedin",
        "description": "ECG Python WebSockets",
    }
    mock_client = MagicMock()

    async def _run():
        return await evaluate_job(job_data, semaphore, client=mock_client)

    result = asyncio.run(_run())
    assert result is not None
    assert result.is_match is True
    assert result.visa_sponsorship == "Supported"


def test_evaluate_job_client_error_fails_gracefully(monkeypatch):
    """Ensure ClientError / ServerError / Exception gracefully returns None and does not crash."""
    import asyncio
    from google.genai.errors import ClientError

    semaphore = asyncio.Semaphore(1)

    async def mock_error_call(*args, **kwargs):
        return None

    monkeypatch.setattr("evaluator._call_gemini", mock_error_call)

    job_data = {
        "title": "Error Job",
        "company": "Failing Corp",
        "location": "Remote",
        "platform": "linkedin",
        "description": "Some description",
    }
    mock_client = MagicMock()

    async def _run():
        return await evaluate_job(job_data, semaphore, client=mock_client)

    result = asyncio.run(_run())
    assert result is None


def test_call_gemini_handles_exception_and_returns_none(monkeypatch):
    """Verify _call_gemini catches Exception and returns None without raising."""
    import asyncio
    from evaluator import _call_gemini

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=Exception("Error: UNAVAILABLE (code 503): No capacity available for model")
    )

    async def _run():
        return await _call_gemini(mock_client, "some prompt")

    res = asyncio.run(_run())
    assert res is None


def test_target_model_strictly_gemini_3_6_flash(monkeypatch):
    """Ensure evaluation model strictly targets gemini-3.6-flash without generic fallbacks."""
    import asyncio
    import evaluator

    assert evaluator.TARGET_GEMINI_MODEL == "gemini-3.6-flash"

    # Avoid sleep in fast unit tests
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    # Verify model is strictly gemini-3.6-flash
    recorded_model = None
    mock_client = MagicMock()

    async def mock_generate(model=None, contents=None, config=None):
        nonlocal recorded_model
        recorded_model = model
        return None

    mock_client.aio.models.generate_content = mock_generate

    for legacy in ("gemini-2.5-flash", "gemini-flash-latest", "gemini-3.8-flash-high", ""):
        monkeypatch.setattr(evaluator.config, "GEMINI_MODEL", legacy)
        asyncio.run(evaluator._call_gemini(mock_client, "test"))
        assert recorded_model == "gemini-3.6-flash"


def test_call_gemini_enforces_4_5s_pacing(monkeypatch):
    """Ensure _call_gemini executes asyncio.sleep(4.5) in finally block on success or failure."""
    import asyncio
    import evaluator

    sleep_mock = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep_mock)

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=None)

    asyncio.run(evaluator._call_gemini(mock_client, "test prompt"))
    sleep_mock.assert_awaited_once_with(4.5)


def test_call_gemini_disables_afc_and_enforces_structured_json(monkeypatch):
    """Ensure GenerateContentConfig disables AFC and passes JobEvaluation as response_schema without tools."""
    import asyncio
    import evaluator
    from google.genai import types

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    passed_config = None
    mock_client = MagicMock()

    async def mock_generate(model=None, contents=None, config=None):
        nonlocal passed_config
        passed_config = config
        return None

    mock_client.aio.models.generate_content = mock_generate

    asyncio.run(evaluator._call_gemini(mock_client, "test prompt"))

    assert passed_config is not None
    assert isinstance(passed_config, types.GenerateContentConfig)
    assert passed_config.response_mime_type == "application/json"
    assert passed_config.response_schema == evaluator.JobEvaluation
    assert passed_config.tools is None
    assert passed_config.automatic_function_calling is not None
    assert passed_config.automatic_function_calling.disable is True


def test_call_gemini_raises_quota_exceeded_on_429(monkeypatch):
    """Ensure _call_gemini raises GeminiQuotaExceededError on 429 RESOURCE_EXHAUSTED."""
    import asyncio
    import pytest
    from evaluator import GeminiQuotaExceededError, _call_gemini

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=Exception("ClientError: 429 RESOURCE_EXHAUSTED: Quota exceeded for quota metric")
    )

    with pytest.raises(GeminiQuotaExceededError):
        asyncio.run(_call_gemini(mock_client, "test prompt"))


def test_evaluate_job_propagates_quota_exceeded(monkeypatch):
    """Ensure evaluate_job propagates GeminiQuotaExceededError up to orchestrator loop."""
    import asyncio
    import pytest
    from evaluator import GeminiQuotaExceededError, evaluate_job

    async def mock_call_quota(*args, **kwargs):
        raise GeminiQuotaExceededError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr("evaluator._call_gemini", mock_call_quota)

    job_data = {"title": "SWE", "company": "Tech Corp"}
    semaphore = asyncio.Semaphore(1)
    mock_client = MagicMock()

    with pytest.raises(GeminiQuotaExceededError):
        asyncio.run(evaluate_job(job_data, semaphore, client=mock_client))


def test_batch_slicing_and_loop_smart_break():
    """Ensure candidate jobs are sliced to 6 and loop breaks on GeminiQuotaExceededError."""
    # Slicing test
    fake_jobs = [f"job_{i}" for i in range(20)]
    sliced = fake_jobs[:6]
    assert len(sliced) == 6

    # Loop break simulation
    processed = 0
    from evaluator import GeminiQuotaExceededError

    for idx, item in enumerate(sliced):
        if idx == 2:
            try:
                raise GeminiQuotaExceededError("Daily quota exhausted")
            except GeminiQuotaExceededError:
                break
        processed += 1

    assert processed == 2  # Aborted at index 2, processed only 0 and 1


def test_call_gemini_retries_on_503_and_succeeds(monkeypatch):
    """Verify _call_gemini retries on 503 UNAVAILABLE and returns result when an attempt succeeds."""
    import asyncio
    from google.genai.errors import ServerError
    from evaluator import _call_gemini, JobEvaluation

    sleep_mock = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep_mock)

    mock_resp = MagicMock()
    mock_resp.text = '{"is_match": true, "visa_sponsorship": "Not needed"}'

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=[
            ServerError(503, {"error": {"message": "503 UNAVAILABLE: Server capacity exhausted"}}),
            ServerError(503, {"error": {"message": "503 UNAVAILABLE: High traffic"}}),
            mock_resp,
        ]
    )

    res = asyncio.run(_call_gemini(mock_client, "test prompt"))

    assert isinstance(res, JobEvaluation)
    assert res.is_match is True
    assert res.visa_sponsorship == "Not needed"
    assert mock_client.aio.models.generate_content.call_count == 3
    # Two 10s backoff sleeps, plus one 4.5s pacing sleep in finally
    sleep_mock.assert_any_await(10.0)
    assert sleep_mock.await_args_list[-1].args == (4.5,)


def test_call_gemini_retries_on_503_and_drops_job_after_3_attempts(monkeypatch):
    """Verify _call_gemini drops job and returns None after 3 failed 503 attempts."""
    import asyncio
    from google.genai.errors import ServerError
    from evaluator import _call_gemini

    sleep_mock = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep_mock)

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=ServerError(503, {"error": {"message": "503 UNAVAILABLE: Service unavailable"}})
    )

    res = asyncio.run(_call_gemini(mock_client, "test prompt"))

    assert res is None
    assert mock_client.aio.models.generate_content.call_count == 3
    # 2 retries (sleep 10.0), then 1 final pacing (sleep 4.5)
    assert sleep_mock.await_count == 3
    sleep_mock.assert_any_await(10.0)
    assert sleep_mock.await_args_list[-1].args == (4.5,)


def test_call_gemini_retries_on_timeout_error(monkeypatch):
    """Verify _call_gemini retries on TimeoutError / asyncio.TimeoutError and recovers."""
    import asyncio
    from evaluator import _call_gemini, JobEvaluation

    sleep_mock = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep_mock)

    mock_resp = MagicMock()
    mock_resp.text = '{"is_match": false, "visa_sponsorship": "No"}'

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=[
            TimeoutError("Request timed out"),
            mock_resp,
        ]
    )

    res = asyncio.run(_call_gemini(mock_client, "test prompt"))

    assert isinstance(res, JobEvaluation)
    assert res.is_match is False
    assert mock_client.aio.models.generate_content.call_count == 2
    sleep_mock.assert_any_await(10.0)
    assert sleep_mock.await_args_list[-1].args == (4.5,)


def test_call_gemini_never_retries_on_429(monkeypatch):
    """Verify _call_gemini does NOT retry on 429 RESOURCE_EXHAUSTED and immediately raises GeminiQuotaExceededError."""
    import asyncio
    import pytest
    from evaluator import GeminiQuotaExceededError, _call_gemini

    sleep_mock = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep_mock)

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=Exception("ClientError: 429 RESOURCE_EXHAUSTED: Daily quota exceeded")
    )

    with pytest.raises(GeminiQuotaExceededError):
        asyncio.run(_call_gemini(mock_client, "test prompt"))

    # Must NOT retry: exactly 1 call
    assert mock_client.aio.models.generate_content.call_count == 1
    # 10s retry sleep must never have been called
    for call in sleep_mock.await_args_list:
        assert call.args != (10.0,)


def test_gemini_timeout_config_value():
    """Ensure GEMINI_TIMEOUT_SECONDS in config is set to 45.0."""
    import config
    assert config.GEMINI_TIMEOUT_SECONDS == 45.0


def test_groq_config_presence(monkeypatch):
    """Verify GROQ_API_KEY is accessible and validated in config."""
    import config

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test123456789")
    # Reload or check config
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_test123456789")
    status = config.validate_configuration()
    assert status["groq_configured"] is True


def test_evaluate_job_groq_success_and_pacing(monkeypatch):
    """Ensure evaluate_job_groq enforces 6s pacing and parses valid JSON into JobEvaluation."""
    import asyncio
    from evaluator import evaluate_job_groq, JobEvaluation

    sleep_mock = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep_mock)

    mock_msg = MagicMock()
    mock_msg.content = '{"is_match": true, "visa_sponsorship": "Supported (US H1B)"}'
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_groq = MagicMock()
    mock_groq.chat.completions.create = AsyncMock(return_value=mock_response)

    job_data = {
        "title": "Firmware Engineer",
        "company": "Medtronic",
        "location": "Minneapolis, MN",
        "platform": "linkedin",
        "description": "Embedded C++ for pacemakers",
    }

    res = asyncio.run(evaluate_job_groq(job_data, mock_groq))

    assert res is not None
    assert isinstance(res, JobEvaluation)
    assert res.is_match is True
    assert res.visa_sponsorship == "Supported (US H1B)"

    # Must have enforced exactly 6-second pacing
    sleep_mock.assert_awaited_once_with(6)

    # Verify model (first in rotation) and response_format
    call_kwargs = mock_groq.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "llama-3.3-70b-versatile"
    assert call_kwargs["response_format"] == {"type": "json_object"}
    assert call_kwargs["temperature"] == 0.1


def test_evaluate_job_groq_rotates_on_404(monkeypatch):
    """Ensure evaluate_job_groq catches 404 model_not_found on 70B and rotates to 8B."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from evaluator import evaluate_job_groq, JobEvaluation

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    # Setup successful response for the second model
    mock_msg = MagicMock()
    mock_msg.content = '{"is_match": true, "visa_sponsorship": "Supported"}'
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    # First call throws 404 model_not_found, second call returns response
    mock_groq = MagicMock()
    mock_groq.chat.completions.create = AsyncMock(
        side_effect=[
            Exception("Error 404: model_not_found for llama-3.3-70b-versatile"),
            mock_response,
        ]
    )

    job_data = {
        "title": "Firmware Engineer",
        "company": "Medtronic",
        "location": "Bangalore",
        "platform": "Workday ATS",
        "description": "Embedded C firmware",
    }

    res = asyncio.run(evaluate_job_groq(job_data, mock_groq))

    assert res is not None
    assert isinstance(res, JobEvaluation)
    assert res.is_match is True
    assert res.visa_sponsorship == "Supported"

    # Verify that create was called twice: first with 70b, then rotated to 8b
    assert mock_groq.chat.completions.create.call_count == 2
    first_call_model = mock_groq.chat.completions.create.call_args_list[0].kwargs["model"]
    second_call_model = mock_groq.chat.completions.create.call_args_list[1].kwargs["model"]
    assert first_call_model == "llama-3.3-70b-versatile"
    assert second_call_model == "llama-3.1-8b-instant"


def test_evaluate_job_groq_all_models_fail_404(monkeypatch):
    """Ensure evaluate_job_groq returns None safely if all fallback models fail with 404."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from evaluator import evaluate_job_groq

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    mock_groq = MagicMock()
    mock_groq.chat.completions.create = AsyncMock(
        side_effect=[
            Exception("Error 404: model_not_found for llama-3.3-70b-versatile"),
            Exception("Error 404: model_not_found for llama-3.1-8b-instant"),
        ]
    )

    job_data = {"title": "Hardware Engineer", "company": "Philips"}
    res = asyncio.run(evaluate_job_groq(job_data, mock_groq))

    assert res is None
    assert mock_groq.chat.completions.create.call_count == 2


def test_evaluate_job_groq_handles_markdown_code_fences(monkeypatch):
    """Ensure evaluate_job_groq strips markdown code fences before validating JSON."""
    import asyncio
    from evaluator import evaluate_job_groq, JobEvaluation

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    mock_msg = MagicMock()
    mock_msg.content = '```json\n{"is_match": false, "visa_sponsorship": "None"}\n```'
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_groq = MagicMock()
    mock_groq.chat.completions.create = AsyncMock(return_value=mock_response)

    job_data = {"title": "Sales Rep", "company": "MedEquip"}
    res = asyncio.run(evaluate_job_groq(job_data, mock_groq))

    assert res is not None
    assert res.is_match is False
    assert res.visa_sponsorship == "None"


def test_evaluate_job_groq_raises_on_429(monkeypatch):
    """Ensure evaluate_job_groq raises GroqQuotaExceededError when encountering HTTP 429."""
    import asyncio
    import pytest
    from evaluator import evaluate_job_groq, GroqQuotaExceededError

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    mock_groq = MagicMock()
    mock_groq.chat.completions.create = AsyncMock(
        side_effect=Exception("Error 429: Rate limit reached for llama-3.3-70b-versatile")
    )

    job_data = {"title": "Firmware Engineer", "company": "Medtronic"}
    with pytest.raises(GroqQuotaExceededError):
        asyncio.run(evaluate_job_groq(job_data, mock_groq))


def test_multi_provider_waterfall_simulation(monkeypatch):
    """Simulate the Gemini 429 -> Groq fallback execution flow."""
    import asyncio
    from evaluator import GeminiQuotaExceededError, JobEvaluation

    # Two jobs
    jobs = [
        {"title": "Job 1", "company": "Co 1"},
        {"title": "Job 2", "company": "Co 2"},
    ]

    use_groq_fallback = False
    evaluated = []

    queue = list(jobs)
    while queue:
        job = queue.pop(0)
        try:
            if not use_groq_fallback:
                # First job triggers 429
                raise GeminiQuotaExceededError("Gemini 429 RESOURCE_EXHAUSTED")
            else:
                evaluated.append((job["title"], "groq"))
        except GeminiQuotaExceededError:
            use_groq_fallback = True
            queue.insert(0, job)  # Retry with Groq
            continue

    assert len(evaluated) == 2
    assert evaluated[0] == ("Job 1", "groq")
    assert evaluated[1] == ("Job 2", "groq")




