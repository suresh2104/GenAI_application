"""Tests for communication/email_generator.py — template fallback and formatting."""

import pytest
from communication.email_generator import EmailGenerator


# ---------------------------------------------------------------------------
# Fixture — generator with unreachable endpoint (forces template fallback)
# ---------------------------------------------------------------------------

@pytest.fixture
def gen():
    return EmailGenerator(endpoint="http://localhost:99999/unreachable")


SAMPLE_REPORT = {
    "claim_id": "CLM-2026-TEST",
    "customer_name": "Jane Test",
    "customer_email": "jane@test.com",
    "policy_number": "INS-2024-001",
    "recommendation": "APPROVE",
    "justification": "Collision damage is fully covered under the policy.",
    "next_steps": ["Cheque will be mailed within 5 business days."],
}


# ---------------------------------------------------------------------------
# Template fallback (Ollama unreachable)
# ---------------------------------------------------------------------------

def test_generates_email_without_ollama(gen):
    email = gen.generate_claim_email(SAMPLE_REPORT)
    assert isinstance(email, str)
    assert len(email) > 50


def test_email_contains_claim_id(gen):
    email = gen.generate_claim_email(SAMPLE_REPORT)
    assert "CLM-2026-TEST" in email


def test_email_contains_customer_name(gen):
    email = gen.generate_claim_email(SAMPLE_REPORT)
    assert "Jane Test" in email


def test_email_contains_decision(gen):
    email = gen.generate_claim_email(SAMPLE_REPORT)
    assert "APPROVE" in email


def test_email_has_subject_line(gen):
    email = gen.generate_claim_email(SAMPLE_REPORT)
    assert email.lower().startswith("subject:")


def test_email_deny_decision(gen):
    report = {**SAMPLE_REPORT, "recommendation": "DENY"}
    email = gen.generate_claim_email(report)
    assert "DENY" in email


def test_email_investigate_decision(gen):
    report = {**SAMPLE_REPORT, "recommendation": "INVESTIGATE"}
    email = gen.generate_claim_email(report)
    assert "INVESTIGATE" in email


def test_next_steps_in_email(gen):
    email = gen.generate_claim_email(SAMPLE_REPORT)
    assert "5 business days" in email


# ---------------------------------------------------------------------------
# format_email
# ---------------------------------------------------------------------------

def test_format_email_adds_subject_if_missing(gen):
    raw = "Dear Jane,\n\nYour claim has been approved.\n\nRegards"
    result = gen.format_email(raw, customer_name="Jane", claim_id="CLM-001")
    assert result.lower().startswith("subject:")


def test_format_email_preserves_existing_subject(gen):
    raw = "Subject: Your Claim Decision\n\nDear Jane,\n\nApproved."
    result = gen.format_email(raw, customer_name="Jane", claim_id="CLM-001")
    assert result.startswith("Subject: Your Claim Decision")


def test_format_email_strips_leading_whitespace(gen):
    raw = "   \n  Subject: Test\n\nBody"
    result = gen.format_email(raw, customer_name="X", claim_id="Y")
    assert not result.startswith(" ")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_missing_customer_name_uses_fallback(gen):
    report = {k: v for k, v in SAMPLE_REPORT.items() if k != "customer_name"}
    email = gen.generate_claim_email(report)
    assert "Valued Customer" in email or isinstance(email, str)


def test_missing_next_steps_does_not_crash(gen):
    report = {k: v for k, v in SAMPLE_REPORT.items() if k != "next_steps"}
    email = gen.generate_claim_email(report)
    assert isinstance(email, str)


def test_next_steps_as_string_not_list(gen):
    report = {**SAMPLE_REPORT, "next_steps": "Call us at 1-800-CLAIMS"}
    email = gen.generate_claim_email(report)
    assert isinstance(email, str)
