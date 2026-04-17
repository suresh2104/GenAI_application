"""Tests for models/schemas.py — Pydantic data model validation."""

import pytest
from pydantic import ValidationError
from models.schemas import PolicyInfo, ClaimRecord, DamageAnalysis, ClaimReport


# ---------------------------------------------------------------------------
# PolicyInfo
# ---------------------------------------------------------------------------

class TestPolicyInfo:
    def _valid(self, **overrides):
        base = dict(
            policy_number="INS-001",
            coverage_types=["fire", "collision"],
            deductible=500.0,
            max_coverage=10000.0,
            exclusions=["war"],
            customer_name="Alice",
            customer_email="alice@example.com",
        )
        return PolicyInfo(**{**base, **overrides})

    def test_valid_policy(self):
        p = self._valid()
        assert p.policy_number == "INS-001"

    def test_empty_coverage_types(self):
        p = self._valid(coverage_types=[])
        assert p.coverage_types == []

    def test_empty_exclusions_default(self):
        p = self._valid(exclusions=[])
        assert p.exclusions == []

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            PolicyInfo(
                policy_number="INS-001",
                deductible=500.0,
                max_coverage=10000.0,
                customer_name="Alice",
                # customer_email missing
            )

    def test_invalid_deductible_type_raises(self):
        with pytest.raises(ValidationError):
            self._valid(deductible="not_a_number")


# ---------------------------------------------------------------------------
# ClaimRecord
# ---------------------------------------------------------------------------

class TestClaimRecord:
    def _valid(self, **overrides):
        base = dict(
            claim_id="CLM-001",
            policy_number="INS-001",
            claim_date="2026-03-25",
            damage_type="collision",
            severity="medium",
            cost_estimate=3200.0,
        )
        return ClaimRecord(**{**base, **overrides})

    def test_valid_record(self):
        r = self._valid()
        assert r.claim_id == "CLM-001"

    def test_default_status_is_pending(self):
        r = self._valid()
        assert r.status == "PENDING"

    def test_default_decision_is_none(self):
        r = self._valid()
        assert r.decision is None

    def test_explicit_decision(self):
        r = self._valid(decision="APPROVED")
        assert r.decision == "APPROVED"

    def test_missing_claim_id_raises(self):
        with pytest.raises(ValidationError):
            ClaimRecord(
                policy_number="INS-001",
                claim_date="2026-01-01",
                damage_type="fire",
                severity="low",
                cost_estimate=500.0,
            )

    def test_invalid_cost_estimate_raises(self):
        with pytest.raises(ValidationError):
            self._valid(cost_estimate="five thousand")


# ---------------------------------------------------------------------------
# DamageAnalysis
# ---------------------------------------------------------------------------

class TestDamageAnalysis:
    def _valid(self, **overrides):
        base = dict(
            damage_type="collision",
            severity="high",
            cost_range="$3000-$6000",
            coverage_eligible=True,
            decision="APPROVE",
            confidence=0.85,
        )
        return DamageAnalysis(**{**base, **overrides})

    def test_valid_analysis(self):
        d = self._valid()
        assert d.confidence == 0.85

    def test_confidence_boundary_zero(self):
        d = self._valid(confidence=0.0)
        assert d.confidence == 0.0

    def test_confidence_boundary_one(self):
        d = self._valid(confidence=1.0)
        assert d.confidence == 1.0

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError):
            self._valid(confidence=1.1)

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValidationError):
            self._valid(confidence=-0.1)

    def test_coverage_eligible_false(self):
        d = self._valid(coverage_eligible=False)
        assert d.coverage_eligible is False


# ---------------------------------------------------------------------------
# ClaimReport
# ---------------------------------------------------------------------------

class TestClaimReport:
    def _make(self):
        policy = PolicyInfo(
            policy_number="INS-001",
            coverage_types=["collision"],
            deductible=500.0,
            max_coverage=10000.0,
            exclusions=[],
            customer_name="Bob",
            customer_email="bob@example.com",
        )
        damage = DamageAnalysis(
            damage_type="collision",
            severity="medium",
            cost_range="$2000-$4000",
            coverage_eligible=True,
            decision="APPROVE",
            confidence=0.9,
        )
        return ClaimReport(
            claim_id="CLM-RPT-001",
            policy_info=policy,
            damage_analysis=damage,
            recommendation="APPROVE",
        )

    def test_valid_report(self):
        r = self._make()
        assert r.claim_id == "CLM-RPT-001"
        assert r.recommendation == "APPROVE"

    def test_email_draft_defaults_empty(self):
        r = self._make()
        assert r.email_draft == ""

    def test_text_analysis_defaults_empty_dict(self):
        r = self._make()
        assert r.text_analysis == {}

    def test_set_email_draft(self):
        r = self._make()
        r.email_draft = "Dear Bob, your claim is approved."
        assert "approved" in r.email_draft
