"""Tests for rag/retriever.py — RAGRetriever context building."""

import pytest
from rag.policy_store import PolicyStore
from rag.claims_db import ClaimsDatabase
from rag.retriever import RAGRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def retriever(tmp_path):
    store = PolicyStore(store_path=str(tmp_path / "vector_store"))
    db = ClaimsDatabase(db_path=str(tmp_path / "claims.db"))
    return RAGRetriever(store, db)


# ---------------------------------------------------------------------------
# retrieve_context — structure
# ---------------------------------------------------------------------------

def test_retrieve_context_returns_required_keys(retriever):
    ctx = retriever.retrieve_context("collision damage", "INS-2024-001")
    assert "policy_chunks" in ctx
    assert "policy_metadata" in ctx
    assert "claims_history" in ctx
    assert "policy_number" in ctx


def test_retrieve_context_policy_number_echoed(retriever):
    ctx = retriever.retrieve_context("fire damage", "INS-2024-002")
    assert ctx["policy_number"] == "INS-2024-002"


def test_retrieve_context_policy_chunks_is_list(retriever):
    ctx = retriever.retrieve_context("flood", "INS-2024-001")
    assert isinstance(ctx["policy_chunks"], list)


def test_retrieve_context_metadata_length_matches_chunks(retriever):
    ctx = retriever.retrieve_context("collision", "INS-2024-001")
    assert len(ctx["policy_chunks"]) == len(ctx["policy_metadata"])


def test_retrieve_context_returns_chunks(retriever):
    ctx = retriever.retrieve_context("collision damage to vehicle", "INS-2024-001")
    assert len(ctx["policy_chunks"]) > 0


def test_retrieve_context_claims_history_is_string(retriever):
    ctx = retriever.retrieve_context("theft", "INS-2024-002")
    assert isinstance(ctx["claims_history"], str)


def test_retrieve_context_unknown_policy_history(retriever):
    ctx = retriever.retrieve_context("anything", "INS-UNKNOWN-999")
    assert "No previous claims" in ctx["claims_history"]


# ---------------------------------------------------------------------------
# build_context_prompt — output format
# ---------------------------------------------------------------------------

def test_build_context_prompt_returns_string(retriever):
    prompt = retriever.build_context_prompt("vehicle fire", "INS-2024-001")
    assert isinstance(prompt, str)


def test_build_context_prompt_contains_sections(retriever):
    prompt = retriever.build_context_prompt("collision", "INS-2024-001")
    assert "RETRIEVED POLICY CONTEXT" in prompt
    assert "CLAIMS HISTORY" in prompt
    assert "USER QUERY" in prompt


def test_build_context_prompt_contains_query(retriever):
    query = "rear-end collision on highway"
    prompt = retriever.build_context_prompt(query, "INS-2024-001")
    assert query in prompt


def test_build_context_prompt_contains_policy_text(retriever):
    prompt = retriever.build_context_prompt("fire damage smoke", "INS-2024-003")
    assert len(prompt) > 100  # meaningful content injected
