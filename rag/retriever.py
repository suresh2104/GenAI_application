"""
RAG retriever – combines policy store and claims history into LLM context.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from rag.policy_store import PolicyStore
from rag.claims_db import ClaimsDatabase

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Orchestrates retrieval from both the policy vector store and the
    claims history database to build context for downstream LLM calls.
    """

    def __init__(self, policy_store: PolicyStore, claims_db: ClaimsDatabase):
        self.policy_store = policy_store
        self.claims_db = claims_db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve_context(self, query: str, policy_number: str) -> Dict[str, Any]:
        """
        Retrieve relevant policy chunks and claims history for *query*.

        Returns a dict with:
          - ``policy_chunks``: list of retrieved policy text snippets
          - ``policy_metadata``: list of metadata dicts from the policy store
          - ``claims_history``: human-readable claim history string
          - ``policy_number``: the policy number used for lookup
        """
        policy_results = self.policy_store.retrieve_policy(query, k=3)

        policy_chunks = [r["text"] for r in policy_results]
        policy_metadata = [r["metadata"] for r in policy_results]

        claims_history = self.claims_db.get_claim_history_summary(policy_number)

        return {
            "policy_chunks": policy_chunks,
            "policy_metadata": policy_metadata,
            "claims_history": claims_history,
            "policy_number": policy_number,
        }

    def build_context_prompt(self, query: str, policy_number: str) -> str:
        """
        Return a single formatted string combining retrieved policy context
        and claims history, ready to inject into an LLM prompt.
        """
        ctx = self.retrieve_context(query, policy_number)

        policy_section = "\n\n".join(
            f"[Policy Document {i + 1}]\n{chunk}"
            for i, chunk in enumerate(ctx["policy_chunks"])
        )

        prompt = (
            "=== RETRIEVED POLICY CONTEXT ===\n"
            f"{policy_section}\n\n"
            "=== CLAIMS HISTORY ===\n"
            f"{ctx['claims_history']}\n\n"
            "=== USER QUERY ===\n"
            f"{query}"
        )
        return prompt
