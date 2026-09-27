"""Tests for Layer 1 Regex Spam Filter and Layer 2 Token Optimization."""

import pytest
from filters import is_spam_title, clean_and_truncate_text


@pytest.mark.parametrize(
    "title,category",
    [
        ("Medical Coding Specialist", "Medical/Clerical"),
        ("Executive Medical Billing Associate", "Medical/Clerical"),
        ("Chief Pharmacist - Retail", "Medical/Clerical"),
        ("Hospital Operations Manager", "Medical/Clerical"),
        ("Senior Sales Executive - MedTech", "Sales/BDE"),
        ("BDE - Healthcare Accounts", "Sales/BDE"),
        ("Business Development Manager", "Sales/BDE"),
        ("Retail Sales Associate", "Sales/BDE"),
        ("Customer Support Representative", "Customer Support"),
        ("Customer Success Associate", "Customer Support"),
        ("Field Service Engineer - MRI", "Field/Maintenance"),
        ("Biomedical Service Technician", "Field/Maintenance"),
        ("Maintenance Engineer - Facilities", "Field/Maintenance"),
        ("AMC Service Coordinator", "Field/Maintenance"),
        ("Calibration Technician", "Hardware"),
        ("Biomedical Technician - Hospital Ward", "Hardware"),
    ],
)
def test_spam_titles_are_detected(title, category):
    is_spam, reason = is_spam_title(title)
    assert is_spam is True
    assert category.lower() in reason.lower()


@pytest.mark.parametrize(
    "title",
    [
        "Biomedical Software Engineer (Python & IoT)",
        "Embedded Systems Developer - Medical Devices",
        "Python Full Stack Engineer - HealthTech",
        "Computer Vision Engineer - Medical Imaging",
        "Cloud Service Architect",  # Must NOT be flagged as Field Service
        "Telemetry Systems Engineer",
        "IoT Firmware Engineer",
    ],
)
def test_legitimate_titles_pass_filter(title):
    is_spam, reason = is_spam_title(title)
    assert is_spam is False
    assert reason == ""


def test_clean_and_truncate_html_stripping():
    raw_html = """
    <html>
        <head>
            <style>body { background: #fff; }</style>
            <script>window.analytics.track('job_view');</script>
        </head>
        <body>
            <h1>Biomedical Python Developer</h1>
            <p>We are seeking a developer with <b>Next.js</b> and <b>WebSockets</b> experience.</p>
        </body>
    </html>
    """
    cleaned = clean_and_truncate_text(raw_html, max_chars=1000)
    assert "analytics" not in cleaned
    assert "background: #fff" not in cleaned
    assert "Biomedical Python Developer" in cleaned
    assert "Next.js" in cleaned
    assert "WebSockets" in cleaned


def test_clean_and_truncate_length_limit():
    long_text = "Python IoT Medical Telemetry " * 500  # ~14,500 chars
    cleaned = clean_and_truncate_text(long_text, max_chars=3500)
    assert len(cleaned) <= 3505
    assert cleaned.endswith("...")


def test_clean_html_text_alias():
    """Verify clean_html_text alias works identically to clean_and_truncate_text."""
    from filters import clean_html_text
    html = "<div><p>Embedded C++ Engineer</p></div>"
    assert clean_html_text(html) == "Embedded C++ Engineer"


def test_semantic_anchoring_differentiates_identical_boilerplate():
    from filters import extract_anchored_description, generate_desc_hash

    # Two jobs sharing the same 600 characters of company boilerplate
    boilerplate = (
        "Welcome to MedTech Enterprise Corp! We are a leading global healthcare technology company "
        "dedicated to improving the lives of millions. We offer comprehensive medical, dental, and vision "
        "benefits, 401(k) matching, paid time off, and continuous learning opportunities. "
        "We are an equal opportunity employer and value diversity across all teams. "
    )
    job1_text = boilerplate + "Key Responsibilities: Design implantable pacemaker firmware using C++."
    job2_text = boilerplate + "Key Responsibilities: Develop Python cloud analytics for hospital telemetry."

    # Anchored extraction should pick up from 'Key Responsibilities:'
    anchored1 = extract_anchored_description(job1_text)
    anchored2 = extract_anchored_description(job2_text)
    assert "implantable pacemaker" in anchored1
    assert "cloud analytics" in anchored2

    # Hashes must differ even though the first 500 characters of the raw text are 100% identical!
    h1 = generate_desc_hash(job1_text)
    h2 = generate_desc_hash(job2_text)
    assert h1 != h2


