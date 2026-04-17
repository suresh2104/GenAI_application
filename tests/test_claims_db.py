"""Tests for rag/claims_db.py — SQLite CRUD and history summary."""

import os
import pytest
from rag.claims_db import ClaimsDatabase


# ---------------------------------------------------------------------------
# Fixture — in-memory / temp DB for each test
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """Fresh ClaimsDatabase backed by a temp file for each test."""
    return ClaimsDatabase(db_path=str(tmp_path / "test_claims.db"))


SAMPLE_CLAIM = {
    "claim_id": "CLM-TEST-001",
    "policy_number": "INS-TEST-001",
    "claim_date": "2026-01-15",
    "damage_type": "collision",
    "severity": "medium",
    "cost_estimate": 3500.0,
    "decision": "APPROVED",
    "status": "CLOSED",
}


# ---------------------------------------------------------------------------
# Initialisation and seeding
# ---------------------------------------------------------------------------

def test_db_creates_and_seeds(db):
    """Fresh DB should be seeded with sample records."""
    claims = db.get_claims_by_policy("INS-2024-001")
    assert len(claims) > 0


def test_db_does_not_reseed_on_second_init(tmp_path):
    """Reopening an existing DB must not duplicate seed data."""
    path = str(tmp_path / "claims.db")
    db1 = ClaimsDatabase(db_path=path)
    count1 = len(db1.get_claims_by_policy("INS-2024-001"))

    db2 = ClaimsDatabase(db_path=path)
    count2 = len(db2.get_claims_by_policy("INS-2024-001"))

    assert count1 == count2


# ---------------------------------------------------------------------------
# add_claim
# ---------------------------------------------------------------------------

def test_add_claim_returns_claim_id(db):
    cid = db.add_claim(SAMPLE_CLAIM)
    assert cid == "CLM-TEST-001"


def test_add_claim_auto_generates_id(db):
    claim = dict(SAMPLE_CLAIM)
    del claim["claim_id"]
    cid = db.add_claim(claim)
    assert cid.startswith("CLM-")
    assert len(cid) > 5


def test_add_claim_defaults_status_to_pending(db):
    claim = {k: v for k, v in SAMPLE_CLAIM.items() if k not in ("status", "decision")}
    claim["claim_id"] = "CLM-TEST-002"
    db.add_claim(claim)
    record = db.get_claim_by_id("CLM-TEST-002")
    assert record.status == "PENDING"


# ---------------------------------------------------------------------------
# get_claim_by_id
# ---------------------------------------------------------------------------

def test_get_claim_by_id_returns_correct_record(db):
    db.add_claim(SAMPLE_CLAIM)
    record = db.get_claim_by_id("CLM-TEST-001")
    assert record is not None
    assert record.damage_type == "collision"
    assert record.cost_estimate == 3500.0


def test_get_claim_by_id_returns_none_for_missing(db):
    assert db.get_claim_by_id("CLM-DOES-NOT-EXIST") is None


# ---------------------------------------------------------------------------
# get_claims_by_policy
# ---------------------------------------------------------------------------

def test_get_claims_by_policy(db):
    db.add_claim(SAMPLE_CLAIM)
    claims = db.get_claims_by_policy("INS-TEST-001")
    assert len(claims) == 1
    assert claims[0].claim_id == "CLM-TEST-001"


def test_get_claims_by_policy_empty_for_unknown(db):
    claims = db.get_claims_by_policy("INS-UNKNOWN-999")
    assert claims == []


def test_get_claims_by_policy_multiple(db):
    claim2 = dict(SAMPLE_CLAIM)
    claim2["claim_id"] = "CLM-TEST-002"
    db.add_claim(SAMPLE_CLAIM)
    db.add_claim(claim2)
    claims = db.get_claims_by_policy("INS-TEST-001")
    assert len(claims) == 2


# ---------------------------------------------------------------------------
# update_claim_status
# ---------------------------------------------------------------------------

def test_update_claim_status(db):
    db.add_claim(SAMPLE_CLAIM)
    result = db.update_claim_status("CLM-TEST-001", "UNDER_REVIEW")
    assert result is True
    record = db.get_claim_by_id("CLM-TEST-001")
    assert record.status == "UNDER_REVIEW"


def test_update_claim_status_with_decision(db):
    db.add_claim(SAMPLE_CLAIM)
    db.update_claim_status("CLM-TEST-001", "CLOSED", decision="DENIED")
    record = db.get_claim_by_id("CLM-TEST-001")
    assert record.decision == "DENIED"
    assert record.status == "CLOSED"


def test_update_nonexistent_claim_returns_false(db):
    result = db.update_claim_status("CLM-GHOST-000", "CLOSED")
    assert result is False


# ---------------------------------------------------------------------------
# get_claim_history_summary
# ---------------------------------------------------------------------------

def test_history_summary_no_claims(db):
    summary = db.get_claim_history_summary("INS-UNKNOWN-999")
    assert "No previous claims" in summary


def test_history_summary_contains_policy_number(db):
    db.add_claim(SAMPLE_CLAIM)
    summary = db.get_claim_history_summary("INS-TEST-001")
    assert "INS-TEST-001" in summary


def test_history_summary_counts(db):
    db.add_claim(SAMPLE_CLAIM)
    denied = dict(SAMPLE_CLAIM)
    denied["claim_id"] = "CLM-TEST-002"
    denied["decision"] = "DENIED"
    db.add_claim(denied)

    summary = db.get_claim_history_summary("INS-TEST-001")
    assert "Approved" in summary or "approved" in summary.lower()
    assert "Denied" in summary or "denied" in summary.lower()
