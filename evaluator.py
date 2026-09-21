"""
Asynchronous LLM Evaluation Engine using Google GenAI SDK (v2).

Features:
  * Strict JSON Schema enforcement via Pydantic model JobEvaluation
  * Exact context-dense system prompt tailored to Biomedical/ECE & Python/IoT portfolio
  * Concurrency control via asyncio.Semaphore(1) and 4.5s pacing to respect free-tier rate limits
  * Single-pass JSON generation with Automatic Function Calling (AFC) disabled
  * Automatic retry handling with tenacity
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional

from google import genai
from google.genai import types
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from google.genai.errors import ServerError
except ImportError:  # pragma: no cover
    class ServerError(Exception):
        pass

import config
from filters import clean_and_truncate_text

log = logging.getLogger(__name__)

# Global Circuit Breaker for Gemini Free Tier
GEMINI_QUOTA_EXHAUSTED = False


class GeminiQuotaExceededError(Exception):
    """Raised when Gemini API daily quota or rate limit is exhausted (429 RESOURCE_EXHAUSTED)."""
    pass


class GroqQuotaExceededError(Exception):
    """Raised when Groq API daily quota or rate limit is exhausted (429)."""
    pass


# --------------------------------------------------------------------------
# A. Structured Output Schema (Pydantic)
# --------------------------------------------------------------------------

class JobEvaluation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    is_match: bool = Field(
        description="Whether this job is a technical match for the candidate's cross-disciplinary profile"
    )
    visa_sponsorship: str = Field(
        default="Not Specified",
        description="Concise extraction of work authorization, visa sponsorship, or relocation support status",
    )
    ai_score: Optional[int] = Field(
        default=None,
        description="Calculated match score between 0 and 100",
    )
    match_reason: str = Field(
        default="",
        description="1-sentence explanation of why this job matches the core priorities or why it was rejected",
    )
    status: str = Field(
        default="active",
        description="Evaluation status: active, consensus_passed, or deferred",
    )
    job_category: str = Field(
        default="General",
        description="Category: Biomedical_RD, ECE_Hardware, Software_Web, or General",
    )

    @property
    def reasoning(self) -> str:
        return self.match_reason

    @property
    def match_score(self) -> Optional[int]:
        return self.ai_score

    @property
    def portfolio_highlight(self) -> str:
        return ""

    @property
    def linkedin_outreach_message(self) -> str:
        return ""


# --------------------------------------------------------------------------
# Resilient JSON Parser
# --------------------------------------------------------------------------

def parse_ai_json(raw_text: str) -> dict[str, Any]:
    """Extracts valid JSON dynamically, ignoring markdown blocks or trailing text."""
    if not raw_text:
        return {}
    cleaned = raw_text.strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(cleaned)
    except Exception:
        return {}


# --------------------------------------------------------------------------
# B. System Prompt (Context-Dense Semantic Matching)
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an elite technical recruiter evaluating engineering roles.

CANDIDATE PROFILE:
The candidate holds a Diploma in Bio-Medical Engineering and a B.Tech in Electronics and Communication Engineering (ECE).
They are a Python Full-Stack Developer and Embedded IoT Engineer. 
Their technical footprint includes:
- Real-time ECG telemetry and arrhythmia detection using WebSockets and React/Next.js.
- Contact-free Human-Computer Interfaces (HCI) using Python, OpenCV, and MediaPipe gesture recognition.
- Embedded hardware systems and IoT water quality monitoring using Arduino and sensors (pH, turbidity, temperature).
- Machine Learning (Data preprocessing, predictive modeling).

YOUR GOAL: Output ONLY a JSON object with keys: is_match (bool), visa_sponsorship (str), ai_score (int), match_reason (str), and job_category (str).

1. CORE PRIORITIES (High ai_score >= 80): 
   Vigorously target R&D roles involving Medical Device development, Biomedical Firmware, IoT sensor integration, Signal Processing, Computer Vision (OpenCV), and Python backend integration with physiological hardware. Boost scores for roles mentioning WebSockets, telemetry, or embedded C/C++.

2. BROAD SEMANTIC MATCHING: 
   Do not reject a job just because the title is generic (e.g., "Software Engineer"). If the company is in MedTech, or the role heavily relies on hardware-software interfacing or Python data pipelines, mark is_match=True.

3. STRICT REJECTIONS (is_match=False): 
   Instantly reject ANY role related to:
   - Sales, Business Development (BDE), Marketing, or Medical Representatives.
   - Field Service, Hardware Maintenance, or Repair Technicians.
   - Medical Billing, Pharmacists, Receptionists, or Clerical hospital staff.
   - IT Helpdesk, Customer Support, or BPO voice processes.

4. MATCH REASON: Provide a 1-sentence explanation linking the job requirements directly to the candidate's profile.

5. JOB CATEGORIZATION: You MUST categorize the job into exactly ONE of the following strings based on its primary focus:
   - "Biomedical_RD" (For MedTech, Medical Devices, Clinical Hardware, ECG telemetry)
   - "ECE_Hardware" (For Embedded C, Arduino, Firmware, Sensors, PCB)
   - "Software_Web" (For Python, React/Next.js, WebSockets, Computer Vision/OpenCV)
   - "General" (For generic roles or if it fits none of the above)
"""

SYSTEM_INSTRUCTION = SYSTEM_PROMPT


# --------------------------------------------------------------------------
# Client Initializer (Cached singleton with timeout configuration)
# --------------------------------------------------------------------------

_client_instance: Optional[genai.Client] = None


def get_genai_client() -> genai.Client:
    global _client_instance
    if _client_instance is not None:
        return _client_instance
    api_key = config.GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured in .env or environment")
    timeout_sec = getattr(config, "GEMINI_TIMEOUT_SECONDS", 45.0)
    _client_instance = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(timeout_sec * 1000)),
    )
    return _client_instance


# --------------------------------------------------------------------------
# Asynchronous Evaluator with Semaphore & Strict Timeout
# --------------------------------------------------------------------------

TARGET_GEMINI_MODEL = "gemini-3.6-flash"


async def _call_gemini(
    client: genai.Client,
    content: str,
) -> Optional[JobEvaluation]:
    """Execute asynchronous generation strictly targeting gemini-3.6-flash.

    Enforces single-pass structured JSON schema matching JobEvaluation,
    disables automatic function calling, and applies 4.5-second pacing delay.
    Retries up to 3 times on transient 503 or timeout errors, but propagates
    429 RESOURCE_EXHAUSTED immediately.

    Args:
        client: Initialized Google GenAI Client instance.
        content: Formatted and token-optimized prompt string.

    Returns:
        JobEvaluation instance if matched/evaluated successfully, or None if dropped.

    Raises:
        GeminiQuotaExceededError: If Google GenAI returns HTTP 429 RESOURCE_EXHAUSTED.
    """
    # Target evaluation model is strictly hardcoded to gemini-3.6-flash
    model_name = TARGET_GEMINI_MODEL

    generation_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=JobEvaluation,
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    max_attempts = 3
    try:
        for attempt in range(1, max_attempts + 1):
            try:
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=content,
                    config=generation_config,
                )
                if not response or not response.text:
                    return None

                return JobEvaluation.model_validate_json(response.text)
            except Exception as exc:
                exc_str = str(exc)
                code = getattr(exc, "code", None)

                # CRITICAL RESTRICTION: Do NOT catch or retry on 429 RESOURCE_EXHAUSTED
                if code == 429 or "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                    log.warning("Gemini 429 RESOURCE_EXHAUSTED (Quota Exceeded): %s", exc)
                    raise GeminiQuotaExceededError(f"Gemini API quota exhausted (429 RESOURCE_EXHAUSTED): {exc}") from exc

                is_retryable = (
                    isinstance(exc, (ServerError, TimeoutError, asyncio.TimeoutError))
                    or code == 503
                    or (isinstance(code, int) and 500 <= code < 600)
                    or "503" in exc_str
                    or "UNAVAILABLE" in exc_str
                    or "timed out" in exc_str.lower()
                    or "timeout" in exc_str.lower()
                )

                if is_retryable:
                    if attempt < max_attempts:
                        log.warning(
                            "Gemini API server/timeout error (attempt %d/%d): %s. Retrying in 10.0s...",
                            attempt,
                            max_attempts,
                            exc,
                        )
                        await asyncio.sleep(10.0)
                        continue
                    else:
                        log.warning(
                            "Gemini API server/timeout error on final attempt (%d/%d): %s. Dropping job from current run.",
                            attempt,
                            max_attempts,
                            exc,
                        )
                        return None

                log.error("Gemini generate_content failed on '%s' (%s): %s", model_name, type(exc).__name__, exc)
                return None
        return None
    finally:
        # Mandatory 4.5-second delay immediately after generate_content execution block (succeeds or fails)
        # Guarantees loop stays strictly under the 10-15 RPM free-tier quota threshold
        await asyncio.sleep(4.5)


def build_job_prompt(job_dict: dict[str, Any]) -> str:
    """Construct unified, token-optimized evaluation prompt from a job record.

    Applies Layer 2 token optimization (stripping HTML, capping text to 3,500 chars)
    to minimize inference latency and stay within provider token budgets.

    Args:
        job_dict: Dictionary containing title, company, location, platform, and description.

    Returns:
        Structured text prompt ready for LLM consumption.
    """
    title = job_dict.get("title", "")
    company = job_dict.get("company", "")
    location = job_dict.get("location", "")
    platform = job_dict.get("platform", "")
    raw_desc = job_dict.get("description", "")

    # Layer 2 Token Optimization applied here
    cleaned_desc = clean_and_truncate_text(raw_desc, max_chars=3500)

    return (
        f"Job Title: {title}\n"
        f"Company: {company}\n"
        f"Location: {location}\n"
        f"Platform: {platform}\n\n"
        f"Job Description:\n{cleaned_desc}"
    )


# Prioritized list of fallback models to attempt in order
GROQ_FALLBACK_MODELS: list[str] = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


async def evaluate_with_gemini(job: dict[str, Any]) -> Optional[JobEvaluation]:
    """Evaluate a single job dictionary with Gemini under a local semaphore."""
    sem = asyncio.Semaphore(1)
    return await evaluate_job(job, sem)


async def evaluate_job_with_model_rotation(
    job_or_prompt: Any,
    groq_client: Any = None,
) -> Any:
    """Evaluate candidate job with 2s pacing and circuit breaker fallback, or rotate Groq models.

    If job_or_prompt is a dict, operates as the top-level paced AI router.
    If job_or_prompt is a str, operates as the internal Groq model rotation handler.
    """
    global GEMINI_QUOTA_EXHAUSTED
    if isinstance(job_or_prompt, dict):
        job = job_or_prompt
        # 2-Second Client-Side Pacing: Protects Groq and OpenRouter from concurrent burst limit 429s
        await asyncio.sleep(2.0)

        # Hard Circuit Breaker Check
        if GEMINI_QUOTA_EXHAUSTED:
            log.info("Gemini circuit breaker open. Instantly routing '%s' to Junior Consensus.", job.get("title", ""))
            return await evaluate_job_consensus(job)

        # Route A (Strict) primary Gemini evaluation
        if job.get("tier") == "strict":
            try:
                eval_res = await evaluate_with_gemini(job)
                if eval_res is not None:
                    return eval_res
                log.warning("Gemini failed for '%s'. Falling back to Junior Consensus.", job.get("title", ""))
                return await evaluate_job_consensus(job)
            except Exception as e:
                error_str = str(e)
                # Detect 429 Quota Exhaustion & Trip Breaker
                if "429" in error_str or "resource_exhausted" in error_str.lower() or "quota" in error_str.lower():
                    log.warning("GEMINI QUOTA EXHAUSTED: Tripping global circuit breaker.")
                    GEMINI_QUOTA_EXHAUSTED = True

                log.warning("Gemini failed for '%s' (%s). Falling back to Junior Consensus.", job.get("title", ""), e)
                return await evaluate_job_consensus(job)

        # Route B (Broad)
        return await evaluate_job_consensus(job)

    prompt = str(job_or_prompt)
    last_exception = None

    for model_name in GROQ_FALLBACK_MODELS:
        try:
            log.info("Attempting Groq evaluation using model: %s", model_name)
            response = await groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"{SYSTEM_INSTRUCTION}\n\n"
                            "You are a strict technical recruiter evaluating biomedical engineering and firmware roles. "
                            "Output ONLY valid JSON matching schema: "
                            "{\"is_match\": boolean, \"visa_sponsorship\": string, \"ai_score\": integer, \"match_reason\": string, \"job_category\": string}."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                model=model_name,
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            return response
        except Exception as e:
            error_str = str(e)
            code = getattr(e, "status_code", getattr(e, "code", None))
            if code == 404 or "404" in error_str or "model_not_found" in error_str:
                log.warning(
                    "Model '%s' returned 404/not found. Rotating to next available model...",
                    model_name,
                )
                last_exception = e
                continue
            elif (
                code == 429
                or "429" in error_str
                or "RESOURCE_EXHAUSTED" in error_str
                or "rate_limit" in error_str.lower()
            ):
                log.warning("Groq API rate limit or quota exceeded (429) on model '%s': %s", model_name, e)
                raise GroqQuotaExceededError(f"Groq API limit reached: {e}") from e
            else:
                # Raise non-404 errors (auth, server, etc.) immediately
                raise e

    raise Exception(f"All Groq fallback models failed. Last error: {last_exception}")


async def evaluate_job_groq(
    job_dict: dict[str, Any],
    groq_client: Any,
) -> Optional[JobEvaluation]:
    """Evaluate a single job posting asynchronously using Groq multi-model fallback rotation with 6s pacing.

    Serves as the secondary stage in the multi-provider waterfall, taking over when
    Gemini hits its 20 daily free requests limit. Attempts prioritized models
    (llama-3.3-70b-versatile -> llama-3.1-8b-instant) and cascades upon 404 model_not_found.
    Strictly enforces a 6-second sleep to maintain throughput below Groq's 12,000 TPM limit.

    Args:
        job_dict: Job dictionary containing title, company, location, platform, description.
        groq_client: Initialized AsyncGroq client instance.

    Returns:
        JobEvaluation instance if evaluated successfully, or None on non-retryable failure.

    Raises:
        GroqQuotaExceededError: If Groq returns HTTP 429 or rate limit exhaustion.
    """
    prompt = build_job_prompt(job_dict)

    log.info("Evaluating with Groq fallback rotation: '%s'", job_dict.get("title", ""))

    # Enforce 6-second pacing to stay under 12,000 TPM limit
    await asyncio.sleep(6)

    try:
        response = await evaluate_job_with_model_rotation(prompt, groq_client)
        if not response or not response.choices:
            return None

        raw_response = response.choices[0].message.content
        if not raw_response:
            return None

        raw_text = raw_response.strip()
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_text = "\n".join(lines).strip()

        evaluation = JobEvaluation.model_validate_json(raw_text)
        log.info(
            "Groq Result for '%s': match=%s",
            job_dict.get("title", ""),
            evaluation.is_match,
        )
        return evaluation
    except GroqQuotaExceededError:
        raise
    except Exception as exc:
        exc_str = str(exc)
        code = getattr(exc, "status_code", getattr(exc, "code", None))
        if code == 429 or "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str or "rate_limit" in exc_str.lower():
            log.warning("Groq API rate limit or quota exceeded (429): %s", exc)
            raise GroqQuotaExceededError(f"Groq API limit reached: {exc}") from exc
        log.error("Groq generate_content failed on '%s': %s", job_dict.get("title", ""), exc)
        return None


async def evaluate_job(
    job_dict: dict[str, Any],
    semaphore: asyncio.Semaphore,
    client: Optional[genai.Client] = None,
) -> Optional[JobEvaluation]:
    """Evaluate a single job posting asynchronously under semaphore throttling and strict timeout.

    Args:
        job_dict: Job dictionary containing title, company, location, platform, description.
        semaphore: asyncio.Semaphore instance (strictly Semaphore(1) for sequential execution).
        client: Optional pre-existing genai.Client instance.

    Returns:
        JobEvaluation instance if evaluated successfully, or None if dropped or timed out.

    Raises:
        GeminiQuotaExceededError: Propagated if Google GenAI returns 429 RESOURCE_EXHAUSTED.
    """
    if client is None:
        client = get_genai_client()

    title = job_dict.get("title", "")
    company = job_dict.get("company", "")
    prompt = build_job_prompt(job_dict)

    global GEMINI_QUOTA_EXHAUSTED
    if GEMINI_QUOTA_EXHAUSTED:
        log.info("Gemini circuit breaker open. Skipping Gemini for '%s' @ %s", title, company)
        raise GeminiQuotaExceededError("Gemini circuit breaker is open (quota previously exhausted)")

    async with semaphore:
        log.info("Evaluating with Gemini: '%s' @ %s", title, company)
        timeout_sec = getattr(config, "GEMINI_TIMEOUT_SECONDS", 45.0)
        try:
            evaluation = await asyncio.wait_for(
                _call_gemini(client, prompt),
                timeout=timeout_sec,
            )
            if evaluation:
                log.info(
                    "Result for '%s': match=%s",
                    title,
                    evaluation.is_match,
                )
            return evaluation
        except (asyncio.TimeoutError, TimeoutError):
            log.warning(
                "Gemini evaluation timed out (%.0fs) for '%s' @ %s — dropping job",
                timeout_sec,
                title,
                company,
            )
            return None
        except GeminiQuotaExceededError:
            GEMINI_QUOTA_EXHAUSTED = True
            # Propagate quota exhaustion so pipeline evaluation loop can break immediately
            raise
        except Exception as exc:
            log.error("Failed evaluation for '%s' @ %s: %s", title, company, exc)
            return None


async def evaluate_jobs_batch(
    jobs: list[dict[str, Any]],
    max_concurrency: int = 1,
) -> list[tuple[dict[str, Any], Optional[JobEvaluation]]]:
    """Evaluate a batch of jobs sequentially with a maximum of 1 concurrent API call."""
    semaphore = asyncio.Semaphore(max_concurrency)
    client = get_genai_client()

    async def _eval_one(j: dict[str, Any]):
        res = await evaluate_job(j, semaphore, client)
        return j, res

    tasks = [_eval_one(job) for job in jobs]
    return await asyncio.gather(*tasks)


# --------------------------------------------------------------------------
# Multi-Provider Consensus Fallback (Groq + OpenRouter)
# --------------------------------------------------------------------------

groq_client = AsyncOpenAI(
    api_key=getattr(config, "GROQ_API_KEY", "") or "mock-key",
    base_url="https://api.groq.com/openai/v1",
)

openrouter_client = AsyncOpenAI(
    api_key=getattr(config, "OPENROUTER_API_KEY", "") or "mock-key",
    base_url="https://openrouter.ai/api/v1",
)


# Reduce stop_after_attempt down to 3 and tune max wait time to 15s to fail gracefully rather than hanging
@retry(wait=wait_exponential(multiplier=1.5, min=2, max=15), stop=stop_after_attempt(3), reraise=False)
async def query_model(client: AsyncOpenAI, model_name: str, prompt: str) -> Optional[dict[str, Any]]:
    """Helper function to query an OpenAI-compatible endpoint safely and parse JSON response.

    Features exponential backoff retry on HTTP 429 rate limit exceptions (up to 3 attempts)
    and resilient regex-based JSON extraction.
    """
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"{SYSTEM_INSTRUCTION}\n\n"
                            "You are a strict technical recruiter evaluating biomedical and firmware roles. "
                            "Output ONLY a valid JSON object containing keys: is_match (bool), visa_sponsorship (str), ai_score (int), match_reason (str), and job_category (str)."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            if not response or not response.choices:
                return None
            content = response.choices[0].message.content or ""
            return parse_ai_json(content)
        except Exception as e:
            exc_str = str(e)
            code = getattr(e, "status_code", getattr(e, "code", None))
            is_rate_limit = code == 429 or "429" in exc_str or "rate_limit" in exc_str.lower()
            if is_rate_limit and attempt < max_attempts:
                backoff = min(15.0, 1.5 ** attempt + 2.0)
                log.warning("Rate limit (429) hit on %s. Retrying in %.2fs (attempt %d/%d)...", model_name, backoff, attempt, max_attempts)
                await asyncio.sleep(backoff)
                continue
            log.error("Error querying model %s: %s", model_name, e)
            return None


async def evaluate_with_consensus(
    prompt: Any,
    groq_cli: Optional[AsyncOpenAI] = None,
    router_cli: Optional[AsyncOpenAI] = None,
) -> dict[str, Any]:
    """Queries Groq (openai/gpt-oss-120b) and OpenRouter concurrently, enforcing strict consensus.

    If models disagree or fail, defaults to deferred status for next-day batch processing.
    """
    if isinstance(prompt, dict):
        prompt = build_job_prompt(prompt)

    g_client = groq_cli or groq_client
    r_client = router_cli or openrouter_client

    log.info("Triggering dual-model consensus fallback (Groq + OpenRouter)...")

    # If keys are missing from runtime environment, defer immediately
    if not getattr(config, "GROQ_API_KEY", ""):
        log.warning("GROQ_API_KEY is not configured. Deferring job evaluation.")
        return {"is_match": False, "visa_sponsorship": "Unknown", "ai_score": 0, "match_reason": "Missing GROQ_API_KEY", "status": "deferred", "job_category": "General"}
    if not getattr(config, "OPENROUTER_API_KEY", ""):
        log.warning("OPENROUTER_API_KEY is not configured. Deferring job evaluation.")
        return {"is_match": False, "visa_sponsorship": "Unknown", "ai_score": 0, "match_reason": "Missing OPENROUTER_API_KEY", "status": "deferred", "job_category": "General"}

    groq_model = getattr(config, "GROQ_ENSEMBLE_MODEL", "openai/gpt-oss-120b")
    router_model = getattr(config, "OPENROUTER_MODEL", "deepseek/deepseek-chat")

    groq_task = query_model(g_client, groq_model, prompt)
    openrouter_task = query_model(r_client, router_model, prompt)

    results = await asyncio.gather(groq_task, openrouter_task, return_exceptions=True)

    groq_res = results[0] if isinstance(results[0], dict) else None
    openrouter_res = results[1] if isinstance(results[1], dict) else None

    # If both models failed to respond correctly
    if not groq_res or not openrouter_res:
        log.warning("One or more ensemble models failed. Deferring job evaluation.")
        return {"is_match": False, "visa_sponsorship": "Unknown", "ai_score": 0, "match_reason": "Ensemble model failure", "status": "deferred", "job_category": "General"}

    groq_match = bool(groq_res.get("is_match", False))
    router_match = bool(openrouter_res.get("is_match", False))
    category = groq_res.get("job_category") or openrouter_res.get("job_category") or "General"

    # 1. Both ACCEPT -> Consensus Passed
    if groq_match and router_match:
        groq_score = groq_res.get("ai_score") or 0
        router_score = openrouter_res.get("ai_score") or 0
        avg_score = (groq_score + router_score) / 2
        return {
            "is_match": True,
            "visa_sponsorship": groq_res.get("visa_sponsorship") or openrouter_res.get("visa_sponsorship") or "Not Specified",
            "ai_score": int(avg_score),
            "match_reason": groq_res.get("match_reason") or openrouter_res.get("match_reason") or "Consensus reached across models.",
            "status": "consensus_passed",
            "job_category": category,
        }

    # 2. Both REJECT -> Consensus Failed (definitive rejection)
    if not groq_match and not router_match:
        log.info("Consensus failed: Both Groq and OpenRouter rejected candidate job.")
        return {
            "is_match": False,
            "visa_sponsorship": "Rejected",
            "ai_score": 0,
            "match_reason": groq_res.get("match_reason") or openrouter_res.get("match_reason") or "Both models rejected role.",
            "status": "consensus_failed",
            "job_category": category,
        }

    # 3. Split Decision -> Conflict (parked for Gemini executive tie-breaker)
    log.info("Consensus conflict: Groq match=%s, OpenRouter match=%s. Deferring as conflict.", groq_match, router_match)
    return {
        "is_match": False,
        "visa_sponsorship": "Conflict",
        "ai_score": 0,
        "match_reason": "Split decision between Groq and OpenRouter.",
        "status": "conflict",
        "job_category": category,
    }


async def evaluate_job_consensus(
    job_dict: dict[str, Any],
    groq_cli: Optional[AsyncOpenAI] = None,
    router_cli: Optional[AsyncOpenAI] = None,
) -> JobEvaluation:
    """Evaluate a single job posting using dual-model consensus fallback."""
    prompt = build_job_prompt(job_dict)
    res_dict = await evaluate_with_consensus(prompt, groq_cli=groq_cli, router_cli=router_cli)
    return JobEvaluation(
        is_match=res_dict.get("is_match", False),
        visa_sponsorship=res_dict.get("visa_sponsorship", "Unknown"),
        ai_score=res_dict.get("ai_score"),
        match_reason=res_dict.get("match_reason", ""),
        status=res_dict.get("status", "deferred"),
        job_category=res_dict.get("job_category", "General"),
    )
