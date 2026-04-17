"""
Pydantic data models (schemas) used across the claim automation pipeline.
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Policy information retrieved from the RAG store / database
# ---------------------------------------------------------------------------

class PolicyInfo(BaseModel):
    """Represents an insurance policy record."""

    policy_number: str = Field(..., description="Unique policy identifier")
    coverage_types: List[str] = Field(
        default_factory=list,
        description="List of covered damage types (e.g. 'collision', 'fire')",
    )
    deductible: float = Field(..., description="Deductible amount in USD")
    max_coverage: float = Field(..., description="Maximum coverage limit in USD")
    exclusions: List[str] = Field(
        default_factory=list,
        description="List of explicitly excluded damage types",
    )
    customer_name: str = Field(..., description="Full name of the policy holder")
    customer_email: str = Field(..., description="Email address of the policy holder")


# ---------------------------------------------------------------------------
# Historical claim record stored in the SQLite database
# ---------------------------------------------------------------------------

class ClaimRecord(BaseModel):
    """Represents a single historical or current insurance claim."""

    claim_id: str = Field(..., description="Unique claim identifier")
    policy_number: str = Field(..., description="Associated policy number")
    claim_date: str = Field(..., description="Date the claim was filed (ISO format)")
    damage_type: str = Field(..., description="Type of damage reported")
    severity: str = Field(
        ..., description="Damage severity: low | medium | high | total_loss"
    )
    cost_estimate: float = Field(..., description="Estimated repair/replacement cost")
    decision: Optional[str] = Field(
        None, description="Final decision: APPROVED | DENIED | PENDING"
    )
    status: str = Field(
        "PENDING", description="Current status: PENDING | CLOSED | UNDER_REVIEW"
    )


# ---------------------------------------------------------------------------
# Output from the vision model damage analysis step
# ---------------------------------------------------------------------------

class DamageAnalysis(BaseModel):
    """Structured result from visual damage assessment."""

    damage_type: str = Field(..., description="Detected type of damage")
    severity: str = Field(
        ..., description="Assessed severity: low | medium | high | total_loss"
    )
    cost_range: str = Field(
        ..., description="Estimated cost range as a human-readable string, e.g. '$2000-$5000'"
    )
    coverage_eligible: bool = Field(
        ..., description="Whether the damage appears eligible under the policy"
    )
    decision: str = Field(
        ..., description="Preliminary decision: APPROVE | DENY | INVESTIGATE"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence score between 0 and 1"
    )


# ---------------------------------------------------------------------------
# Final composite claim report
# ---------------------------------------------------------------------------

class ClaimReport(BaseModel):
    """Aggregated report produced at the end of the full pipeline."""

    claim_id: str = Field(..., description="Unique claim identifier for this report")
    policy_info: PolicyInfo = Field(..., description="Policy details")
    damage_analysis: DamageAnalysis = Field(
        ..., description="Results from visual inspection"
    )
    text_analysis: dict = Field(
        default_factory=dict,
        description="Results from accident-report text analysis",
    )
    recommendation: str = Field(
        ..., description="Final recommendation: APPROVE | DENY | INVESTIGATE"
    )
    email_draft: str = Field(
        "", description="AI-generated customer communication email"
    )
