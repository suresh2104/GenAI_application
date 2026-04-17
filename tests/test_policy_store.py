"""Tests for rag/policy_store.py — vector store and keyword fallback search."""

import pytest
from rag.policy_store import PolicyStore


# ---------------------------------------------------------------------------
# Fixture — fresh store per test (no persisted index)
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """PolicyStore backed by a temp directory so tests don't share state."""
    return PolicyStore(store_path=str(tmp_path / "vector_store"))


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_store_initialises_with_sample_policies(store):
    assert len(store.documents) > 0


def test_store_loads_sample_metadata(store):
    assert len(store.metadata) == len(store.documents)


# ---------------------------------------------------------------------------
# add_policy
# ---------------------------------------------------------------------------

def test_add_policy_increases_document_count(store):
    before = len(store.documents)
    store.add_policy("Test policy text for hail damage.", {"policy_number": "INS-X"})
    assert len(store.documents) == before + 1


def test_add_policy_stores_metadata(store):
    meta = {"policy_number": "INS-NEW", "coverage_types": ["hail"]}
    store.add_policy("Hail damage is covered.", meta)
    assert store.metadata[-1]["policy_number"] == "INS-NEW"


# ---------------------------------------------------------------------------
# retrieve_policy — result structure
# ---------------------------------------------------------------------------

def test_retrieve_returns_list(store):
    results = store.retrieve_policy("collision damage")
    assert isinstance(results, list)


def test_retrieve_returns_dicts_with_required_keys(store):
    results = store.retrieve_policy("fire damage")
    for r in results:
        assert "text" in r
        assert "metadata" in r
        assert "score" in r


def test_retrieve_respects_k_limit(store):
    results = store.retrieve_policy("collision", k=2)
    assert len(results) <= 2


def test_retrieve_k_larger_than_docs_capped(store):
    results = store.retrieve_policy("anything", k=9999)
    assert len(results) <= len(store.documents)


# ---------------------------------------------------------------------------
# retrieve_policy — relevance (keyword fallback always active as a baseline)
# ---------------------------------------------------------------------------

def test_collision_query_returns_collision_docs(store):
    results = store.retrieve_policy("car collision road accident", k=3)
    texts = " ".join(r["text"].lower() for r in results)
    assert "collision" in texts


def test_fire_query_returns_fire_docs(store):
    results = store.retrieve_policy("fire damage arson smoke", k=3)
    texts = " ".join(r["text"].lower() for r in results)
    assert "fire" in texts


def test_water_query_returns_water_docs(store):
    results = store.retrieve_policy("flooding burst pipe water damage", k=3)
    texts = " ".join(r["text"].lower() for r in results)
    assert "water" in texts


# ---------------------------------------------------------------------------
# Empty query edge case
# ---------------------------------------------------------------------------

def test_empty_query_does_not_crash(store):
    results = store.retrieve_policy("", k=3)
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Persistence — index saved and reloaded
# ---------------------------------------------------------------------------

def test_index_persists_across_instances(tmp_path):
    path = str(tmp_path / "vector_store")
    store1 = PolicyStore(store_path=path)
    initial_count = len(store1.documents)

    store1.add_policy("Extra policy about earthquake coverage.", {"policy_number": "INS-EQ"})

    store2 = PolicyStore(store_path=path)
    assert len(store2.documents) == initial_count + 1
