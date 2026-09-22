"""
Standalone Outreach Generator CLI Utility.

Generates high-impact, tailored cold outreach emails and LinkedIn connection notes
for evaluated jobs stored in Turso Cloud Database using Groq LLM (llama-3.3-70b-versatile).

Strictly decoupled from main.py to prevent token quota exhaustion during automated CI/CD runs.

Usage:
    python outreach.py --job-id <job_id> [--copy]
    python outreach.py --recent [limit]
    python outreach.py --search <keyword>
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any, Optional

from openai import AsyncOpenAI

import config
from database import get_job, get_turso_client, parse_turso_rows


OUTREACH_SYSTEM_PROMPT = """You are an elite career strategist and software/biomedical engineering candidate drafting a concise, high-impact cold outreach message (email or LinkedIn note) to an engineering hiring manager.

Candidate Profile:
- Core Competencies: Biomedical Engineering, Medical Device Systems, Embedded Firmware (C/C++), Python Full Stack, Telemetry, and Cloud Systems.
- Engineering Approach: Fast-paced prototyping, safety-critical systems, ISO 13485 / IEC 62304 awareness, real-time telemetry.

Constraints:
1. Subject line must be punchy, relevant, and mention the company and specific engineering problem.
2. Hook: Open with a specific technical detail or requirement extracted directly from their job description.
3. Value Proposition: 1-2 sharp sentences connecting candidate capabilities to their exact project needs.
4. Call to Action (CTA): Low friction (e.g. 10-minute virtual coffee / intro conversation).
5. Length: Under 150 words total. No hollow flattery, no buzzword fluff.

Output Format:
SUBJECT: <Compelling Subject Line>

Hi <Hiring Manager / Team>,

<Body Paragraph 1: Hook referencing specific challenge/need from role>

<Body Paragraph 2: Core technical fit & relevant achievements>

<Body Paragraph 3: Low-friction CTA>

Best regards,
Deepak
"""


async def list_recent_jobs(limit: int = 10) -> None:
    """Display the top recent high-scoring jobs from Turso Cloud Database."""
    client = get_turso_client()
    try:
        query = (
            "SELECT job_id, title, company, location, ai_score, job_category, date_found "
            "FROM job_postings "
            "WHERE ai_score >= 70 "
            "ORDER BY date_found DESC LIMIT ?"
        )
        res = await client.execute_query(query, [limit])
        rows = parse_turso_rows(res)
        if not rows:
            print("\nNo high-match jobs (score >= 70) found in Turso database.")
            return

        print("\n" + "=" * 80)
        print(f" [RECENT HIGH-MATCH JOBS (Top {len(rows)})]")
        print("=" * 80)
        for r in rows:
            jid = r.get("job_id", "N/A")
            title = r.get("title", "Unknown")
            comp = r.get("company", "Unknown")
            score = r.get("ai_score", "0")
            cat = r.get("job_category", "General")
            date = (r.get("date_found") or "")[:10]
            print(f" * ID: {jid} | Score: {score}/100 | [{cat}]")
            print(f"   Role: {title} @ {comp} ({date})")
            print(f"   Command: python outreach.py --job-id {jid}")
            print("-" * 80)
    finally:
        if client.session:
            await client.session.close()


async def search_jobs(query_text: str, limit: int = 10) -> None:
    """Search jobs by title or company in Turso Cloud Database."""
    client = get_turso_client()
    try:
        sql = (
            "SELECT job_id, title, company, location, ai_score, date_found "
            "FROM job_postings "
            "WHERE title LIKE ? OR company LIKE ? "
            "ORDER BY date_found DESC LIMIT ?"
        )
        pattern = f"%{query_text}%"
        res = await client.execute_query(sql, [pattern, pattern, limit])
        rows = parse_turso_rows(res)
        if not rows:
            print(f"\nNo jobs matching '{query_text}' found in database.")
            return

        print("\n" + "=" * 80)
        print(f" [SEARCH RESULTS FOR '{query_text}' ({len(rows)} matches)]")
        print("=" * 80)
        for r in rows:
            jid = r.get("job_id", "N/A")
            title = r.get("title", "Unknown")
            comp = r.get("company", "Unknown")
            score = r.get("ai_score", "0")
            print(f" * ID: {jid} | Score: {score}/100 | {title} @ {comp}")
            print(f"   Command: python outreach.py --job-id {jid}")
            print("-" * 80)
    finally:
        if client.session:
            await client.session.close()


async def generate_outreach_email(
    job_id: str,
    model_name: str = "llama-3.3-70b-versatile",
) -> Optional[str]:
    """Retrieve job by ID from Turso and generate tailored cold email via Groq."""
    client = get_turso_client()
    try:
        job = await get_job(job_id, client=client)
    finally:
        if client.session:
            await client.session.close()

    if not job:
        print(f"[!] Error: Job with ID '{job_id}' not found in Turso database.")
        return None

    title = job.get("title", "Engineering Role")
    company = job.get("company", "Company")
    location = job.get("location", "Not Specified")
    description = job.get("description", "") or "No detailed description available."
    category = job.get("job_category", "General")
    score = job.get("ai_score", 0)

    print(f"\nTargeting: {title} @ {company} (Score: {score}/100, Category: {category})")
    print(f"Generating personalized cold outreach via Groq ({model_name})...\n")

    groq_api_key = getattr(config, "GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
    if not groq_api_key or groq_api_key == "mock-key":
        print("[!] Error: GROQ_API_KEY is not set in .env or environment.")
        return None

    groq_client = AsyncOpenAI(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    user_prompt = f"""Generate a tailored cold outreach email for this job:
Title: {title}
Company: {company}
Location: {location}
Job Category: {category}

Job Description:
{description[:3000]}
"""

    try:
        response = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": OUTREACH_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=model_name,
            temperature=0.4,
            max_tokens=450,
        )
        if response and response.choices:
            return response.choices[0].message.content or ""
        return None
    except Exception as exc:
        print(f"[!] Groq outreach generation failed with {model_name}: {exc}")
        print("Retrying with fast fallback model (llama-3.1-8b-instant)...")
        try:
            response = await groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": OUTREACH_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model="llama-3.1-8b-instant",
                temperature=0.4,
                max_tokens=450,
            )
            if response and response.choices:
                return response.choices[0].message.content or ""
        except Exception as fallback_exc:
            print(f"[!] Fallback generation also failed: {fallback_exc}")
        return None


def copy_to_clipboard(text: str) -> bool:
    """Copies text to the system clipboard if pyperclip is available."""
    try:
        import pyperclip  # type: ignore
        pyperclip.copy(text)
        return True
    except ImportError:
        try:
            # Fallback for Windows clip command
            import subprocess
            process = subprocess.Popen(["clip"], stdin=subprocess.PIPE, close_fds=True)
            process.communicate(input=text.encode("utf-8"))
            return True
        except Exception:
            return False
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Job Intelligence — Cold Outreach Generator (Groq LLM)",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        help="Unique 16-character job ID from Turso database to generate outreach for.",
    )
    parser.add_argument(
        "--recent",
        nargs="?",
        const=10,
        type=int,
        help="List top recent high-match jobs (default: 10).",
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search for jobs by title or company name in Turso.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Automatically copy the generated outreach text to system clipboard.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.3-70b-versatile",
        help="Groq model to use (default: llama-3.3-70b-versatile).",
    )

    args = parser.parse_args()

    if args.recent is not None:
        asyncio.run(list_recent_jobs(limit=args.recent))
        return

    if args.search:
        asyncio.run(search_jobs(query_text=args.search))
        return

    if args.job_id:
        outreach_text = asyncio.run(generate_outreach_email(args.job_id, model_name=args.model))
        if outreach_text:
            print("=" * 70)
            print(outreach_text)
            print("=" * 70)

            if args.copy:
                copied = copy_to_clipboard(outreach_text)
                if copied:
                    print("\n[OK] Outreach message successfully copied to clipboard!")
                else:
                    print("\n[!] Could not copy to clipboard (install pyperclip or use terminal copy).")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
