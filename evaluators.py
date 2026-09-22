"""
Evaluators module alias for the Unified Autonomous AI Job Pipeline.

Re-exports all evaluator routines, fallback consensus, and the Groq batch pre-filter.
"""

from __future__ import annotations

from evaluator import (
    GEMINI_QUOTA_EXHAUSTED,
    GeminiQuotaExceededError,
    JobEvaluation,
    TARGET_GEMINI_MODEL,
    batch_coarse_filter_groq,
    build_job_prompt,
    evaluate_job,
    evaluate_job_consensus,
    evaluate_job_groq,
    evaluate_jobs_batch,
    evaluate_with_consensus,
    get_genai_client,
    groq_client,
    openrouter_client,
    parse_ai_json,
    query_model,
    query_openrouter,
)

__all__ = [
    "GEMINI_QUOTA_EXHAUSTED",
    "GeminiQuotaExceededError",
    "JobEvaluation",
    "TARGET_GEMINI_MODEL",
    "batch_coarse_filter_groq",
    "build_job_prompt",
    "evaluate_job",
    "evaluate_job_consensus",
    "evaluate_job_groq",
    "evaluate_jobs_batch",
    "evaluate_with_consensus",
    "get_genai_client",
    "groq_client",
    "openrouter_client",
    "parse_ai_json",
    "query_model",
    "query_openrouter",
]
