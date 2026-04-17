"""
SQLite-backed claims database.

Provides CRUD operations and historical summary generation for claim records.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from models.schemas import ClaimRecord

logger = logging.getLogger(__name__)


class ClaimsDatabase:
    """Manages insurance claim records in a local SQLite database."""

    CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS claims (
            claim_id      TEXT PRIMARY KEY,
            policy_number TEXT NOT NULL,
            claim_date    TEXT NOT NULL,
            damage_type   TEXT NOT NULL,
            severity      TEXT NOT NULL,
            cost_estimate REAL NOT NULL,
            decision      TEXT,
            status        TEXT NOT NULL DEFAULT 'PENDING'
        )
    """

    def __init__(self, db_path: str = "./data/claims.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.initialize_db()

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_db(self) -> None:
        """Create the claims table if it does not exist and seed sample data."""
        with self._connect() as conn:
            conn.execute(self.CREATE_TABLE_SQL)
            conn.commit()

        # Seed only if table is empty
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]

        if count == 0:
            self._seed_sample_data()
            logger.info("Claims database seeded with sample records.")
        else:
            logger.info("Claims database loaded (%d existing records).", count)

    def _seed_sample_data(self) -> None:
        sample_claims = [
            {
                "claim_id": "CLM-2023-001",
                "policy_number": "INS-2024-001",
                "claim_date": "2023-03-15",
                "damage_type": "collision",
                "severity": "medium",
                "cost_estimate": 3200.0,
                "decision": "APPROVED",
                "status": "CLOSED",
            },
            {
                "claim_id": "CLM-2023-002",
                "policy_number": "INS-2024-001",
                "claim_date": "2023-09-22",
                "damage_type": "water damage",
                "severity": "low",
                "cost_estimate": 850.0,
                "decision": "APPROVED",
                "status": "CLOSED",
            },
            {
                "claim_id": "CLM-2024-001",
                "policy_number": "INS-2024-002",
                "claim_date": "2024-01-10",
                "damage_type": "theft",
                "severity": "high",
                "cost_estimate": 15000.0,
                "decision": "APPROVED",
                "status": "CLOSED",
            },
            {
                "claim_id": "CLM-2024-002",
                "policy_number": "INS-2024-002",
                "claim_date": "2024-06-30",
                "damage_type": "collision",
                "severity": "low",
                "cost_estimate": 1200.0,
                "decision": "DENIED",
                "status": "CLOSED",
            },
            {
                "claim_id": "CLM-2024-003",
                "policy_number": "INS-2024-003",
                "claim_date": "2024-11-05",
                "damage_type": "fire",
                "severity": "high",
                "cost_estimate": 22000.0,
                "decision": "APPROVED",
                "status": "CLOSED",
            },
        ]
        for claim in sample_claims:
            self.add_claim(claim)

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def add_claim(self, claim_record: Dict[str, Any]) -> str:
        """
        Insert a new claim record.

        If *claim_id* is absent it is auto-generated.
        Returns the claim_id of the inserted record.
        """
        if "claim_id" not in claim_record or not claim_record["claim_id"]:
            claim_record = dict(claim_record)  # don't mutate caller's dict
            claim_record["claim_id"] = f"CLM-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

        sql = """
            INSERT OR REPLACE INTO claims
                (claim_id, policy_number, claim_date, damage_type, severity,
                 cost_estimate, decision, status)
            VALUES
                (:claim_id, :policy_number, :claim_date, :damage_type, :severity,
                 :cost_estimate, :decision, :status)
        """
        defaults = {"decision": None, "status": "PENDING"}
        row = {**defaults, **claim_record}
        with self._connect() as conn:
            conn.execute(sql, row)
            conn.commit()
        logger.debug("Claim %s saved to database.", claim_record["claim_id"])
        return claim_record["claim_id"]

    def get_claims_by_policy(self, policy_number: str) -> List[ClaimRecord]:
        """Return all claim records associated with *policy_number*."""
        sql = "SELECT * FROM claims WHERE policy_number = ? ORDER BY claim_date DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, (policy_number,)).fetchall()
        return [ClaimRecord(**dict(row)) for row in rows]

    def get_claim_by_id(self, claim_id: str) -> Optional[ClaimRecord]:
        """Return a single claim record by ID, or None if not found."""
        sql = "SELECT * FROM claims WHERE claim_id = ?"
        with self._connect() as conn:
            row = conn.execute(sql, (claim_id,)).fetchone()
        if row is None:
            return None
        return ClaimRecord(**dict(row))

    def update_claim_status(self, claim_id: str, status: str, decision: Optional[str] = None) -> bool:
        """Update status (and optionally decision) for an existing claim."""
        if decision is not None:
            sql = "UPDATE claims SET status = ?, decision = ? WHERE claim_id = ?"
            params = (status, decision, claim_id)
        else:
            sql = "UPDATE claims SET status = ? WHERE claim_id = ?"
            params = (status, claim_id)

        with self._connect() as conn:
            cur = conn.execute(sql, params)
            conn.commit()

        if cur.rowcount == 0:
            logger.warning("update_claim_status: claim_id %s not found.", claim_id)
            return False
        logger.info("Claim %s updated – status=%s, decision=%s", claim_id, status, decision)
        return True

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_claim_history_summary(self, policy_number: str) -> str:
        """Return a human-readable text summary of the claim history for a policy."""
        claims = self.get_claims_by_policy(policy_number)
        if not claims:
            return f"No previous claims found for policy {policy_number}."

        total = len(claims)
        approved = sum(1 for c in claims if c.decision == "APPROVED")
        denied = sum(1 for c in claims if c.decision == "DENIED")
        total_paid = sum(c.cost_estimate for c in claims if c.decision == "APPROVED")

        lines = [
            f"Claim history for policy {policy_number}:",
            f"  Total claims filed : {total}",
            f"  Approved           : {approved}",
            f"  Denied             : {denied}",
            f"  Total paid out     : ${total_paid:,.2f}",
            "",
            "Recent claims (up to 5):",
        ]
        for claim in claims[:5]:
            lines.append(
                f"  [{claim.claim_date}] {claim.damage_type} – {claim.severity} severity – "
                f"${claim.cost_estimate:,.2f} – {claim.decision or 'PENDING'}"
            )
        return "\n".join(lines)
