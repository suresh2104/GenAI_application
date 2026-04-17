"""Tests for formatter.py — LLaVA prompt construction."""

import pytest
from PIL import Image
from formatter import format_llava_input


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_image(color="red"):
    return Image.new("RGB", (100, 100), color=color)


SAMPLE_POLICY = {
    "policy_number": "INS-001",
    "customer_name": "Test User",
    "coverage_types": ["collision", "fire"],
    "deductible": 500,
    "max_coverage": 10000,
    "exclusions": ["war"],
}


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

def test_returns_list(make_image=make_image):
    result = format_llava_input([make_image()])
    assert isinstance(result, list)


def test_one_prompt_per_image():
    images = [make_image("red"), make_image("blue"), make_image("green")]
    result = format_llava_input(images)
    assert len(result) == 3


def test_each_prompt_has_prompt_and_image_keys():
    result = format_llava_input([make_image()])
    assert "prompt" in result[0]
    assert "image" in result[0]


def test_image_object_stored_in_prompt(make_image=make_image):
    img = make_image()
    result = format_llava_input([img])
    assert result[0]["image"] is img


# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------

def test_prompt_contains_image_tag():
    result = format_llava_input([make_image()])
    assert "<image>" in result[0]["prompt"]


def test_prompt_asks_for_json_output():
    result = format_llava_input([make_image()])
    assert "JSON" in result[0]["prompt"] or "json" in result[0]["prompt"].lower()


def test_prompt_contains_required_keys_hint():
    result = format_llava_input([make_image()])
    prompt = result[0]["prompt"]
    assert "damage_type" in prompt
    assert "decision" in prompt


# ---------------------------------------------------------------------------
# Policy injection
# ---------------------------------------------------------------------------

def test_policy_context_injected_when_provided():
    result = format_llava_input([make_image()], policy=SAMPLE_POLICY)
    assert "INS-001" in result[0]["prompt"] or "INSURANCE POLICY CONTEXT" in result[0]["prompt"]


def test_no_policy_context_without_policy():
    result = format_llava_input([make_image()])
    assert "INSURANCE POLICY CONTEXT" not in result[0]["prompt"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_image_list_returns_empty():
    result = format_llava_input([])
    assert result == []


def test_text_data_does_not_crash():
    result = format_llava_input([make_image()], text_data="Some extracted PDF text")
    assert len(result) == 1
