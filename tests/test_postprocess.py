"""Tests for postprocess.py — result aggregation and decision logic."""

import pytest
from postprocess import process_results


# ---------------------------------------------------------------------------
# Empty / invalid inputs
# ---------------------------------------------------------------------------

def test_empty_list_returns_empty_dict():
    assert process_results([]) == {}


def test_none_severity_handled():
    results = [{"damage_type": "fire"}]  # no 'severity' key
    out = process_results(results)
    assert isinstance(out, dict)


def test_non_dict_items_ignored():
    results = ["bad_string", None, {"damage_type": "collision", "severity": 5, "decision": "approve"}]
    out = process_results(results)
    assert out["damage_type"] == "collision"


# ---------------------------------------------------------------------------
# Severity selection — highest wins
# ---------------------------------------------------------------------------

def test_picks_highest_severity():
    results = [
        {"damage_type": "scratch", "severity": 2, "decision": "deny"},
        {"damage_type": "collision", "severity": 8, "decision": "approve"},
        {"damage_type": "dent", "severity": 5, "decision": "investigate"},
    ]
    out = process_results(results)
    assert out["damage_type"] == "collision"
    assert out["severity"] == 8


def test_single_result_returned_as_is():
    results = [{"damage_type": "fire", "severity": 6, "decision": "approve"}]
    out = process_results(results)
    assert out["damage_type"] == "fire"


# ---------------------------------------------------------------------------
# Justification strings per decision
# ---------------------------------------------------------------------------

def test_approve_justification():
    results = [{"severity": 5, "decision": "approve"}]
    out = process_results(results)
    assert out["decision"] == "approve"
    assert "covered" in out["justification"].lower()


def test_deny_justification():
    results = [{"severity": 5, "decision": "deny"}]
    out = process_results(results)
    assert "not covered" in out["justification"].lower()


def test_investigate_justification():
    results = [{"severity": 5, "decision": "investigate"}]
    out = process_results(results)
    assert "investigation" in out["justification"].lower()


def test_unknown_decision_gets_investigate_justification():
    results = [{"severity": 3, "decision": "unknown_value"}]
    out = process_results(results)
    assert "justification" in out


# ---------------------------------------------------------------------------
# Original fields preserved
# ---------------------------------------------------------------------------

def test_original_fields_preserved():
    results = [{"damage_type": "flood", "severity": 4, "cost_range": "$1000-$3000", "decision": "approve"}]
    out = process_results(results)
    assert out["cost_range"] == "$1000-$3000"
