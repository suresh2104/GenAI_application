"""
Text analyzer – uses Ollama LLM to analyze accident reports and
cross-check findings against retrieved policy context.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


class TextAnalyzer:
    """
    Sends accident report text to an Ollama-hosted LLM and returns
    structured analysis as Python dicts.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:11434/api/generate",
        model_name: str = "deepseek-r1:1.5b",
        timeout: int = 60,
    ):
        self.endpoint = endpoint
        self.model_name = model_name
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str) -> str:
        """Send *prompt* to Ollama and return the raw response string."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        try:
            resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.exceptions.ConnectionError:
            logger.error("Ollama not reachable at %s.", self.endpoint)
            return ""
        except Exception as exc:
            logger.error("Ollama call failed: %s", exc)
            return ""

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Extract the first JSON object found in *text*."""
        # Strip <think>...</think> blocks produced by DeepSeek-R1
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"raw_response": text, "parse_error": True}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_accident_report(
        self, report_text: str, policy_context: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze a free-text accident report with the aid of retrieved
        policy context.

        Returns a dict with keys:
          incident_type, fault_determination, injuries_reported,
          police_report_filed, estimated_damage, key_facts, red_flags
        """
        prompt = (
            "You are an experienced insurance claims analyst.\n\n"
            "POLICY CONTEXT:\n"
            f"{policy_context}\n\n"
            "ACCIDENT REPORT:\n"
            f"{report_text}\n\n"
            "Analyze the accident report above and respond ONLY with a valid JSON object "
            "containing these keys:\n"
            "  incident_type        – brief type of incident (e.g. 'rear-end collision')\n"
            "  fault_determination  – 'insured', 'third_party', 'shared', or 'undetermined'\n"
            "  injuries_reported    – true or false\n"
            "  police_report_filed  – true or false\n"
            "  estimated_damage     – estimated USD repair cost as an integer\n"
            "  key_facts            – list of 3-5 key facts extracted from the report\n"
            "  red_flags            – list of any suspicious or inconsistent details "
            "(empty list if none)\n\n"
            "Respond with the JSON object only. No explanation."
        )
        raw = self._call_ollama(prompt)
        if not raw:
            return {
                "incident_type": "unknown",
                "fault_determination": "undetermined",
                "injuries_reported": False,
                "police_report_filed": False,
                "estimated_damage": 0,
                "key_facts": [],
                "red_flags": [],
                "error": "Ollama unavailable",
            }
        return self._extract_json(raw)

    def cross_check_policy(
        self, damage_analysis: Dict[str, Any], policy_context: str
    ) -> Dict[str, Any]:
        """
        Cross-check the vision-model damage analysis against retrieved
        policy terms.

        Returns a dict with keys:
          coverage_eligible, coverage_reason, exclusions_triggered,
          deductible_applies, recommendation
        """
        damage_summary = json.dumps(damage_analysis, indent=2)
        prompt = (
            "You are an insurance underwriting expert.\n\n"
            "POLICY CONTEXT:\n"
            f"{policy_context}\n\n"
            "DAMAGE ANALYSIS FROM VISUAL INSPECTION:\n"
            f"{damage_summary}\n\n"
            "Determine whether this damage is covered under the policy and respond ONLY "
            "with a valid JSON object containing:\n"
            "  coverage_eligible    – true or false\n"
            "  coverage_reason      – one-sentence explanation\n"
            "  exclusions_triggered – list of policy exclusions that apply (empty if none)\n"
            "  deductible_applies   – true or false\n"
            "  recommendation       – 'APPROVE', 'DENY', or 'INVESTIGATE'\n\n"
            "Respond with the JSON object only. No explanation."
        )
        raw = self._call_ollama(prompt)
        if not raw:
            return {
                "coverage_eligible": False,
                "coverage_reason": "Could not determine – Ollama unavailable",
                "exclusions_triggered": [],
                "deductible_applies": True,
                "recommendation": "INVESTIGATE",
                "error": "Ollama unavailable",
            }
        return self._extract_json(raw)

    def generate_recommendation(
        self,
        damage_analysis: Dict[str, Any],
        text_analysis: Dict[str, Any],
        claims_history: str,
    ) -> Dict[str, Any]:
        """
        Synthesize damage analysis, text analysis, and claims history into
        a final recommendation.

        Returns a dict with keys:
          final_decision, confidence_score, justification, next_steps
        """
        prompt = (
            "You are a senior insurance claims manager making a final decision.\n\n"
            "VISUAL DAMAGE ANALYSIS:\n"
            f"{json.dumps(damage_analysis, indent=2)}\n\n"
            "ACCIDENT REPORT ANALYSIS:\n"
            f"{json.dumps(text_analysis, indent=2)}\n\n"
            "CLAIMS HISTORY:\n"
            f"{claims_history}\n\n"
            "Based on all the above information, provide a final claim recommendation "
            "as a valid JSON object with:\n"
            "  final_decision   – 'APPROVE', 'DENY', or 'INVESTIGATE'\n"
            "  confidence_score – float between 0.0 (low) and 1.0 (high)\n"
            "  justification    – 2-3 sentence explanation of the decision\n"
            "  next_steps       – list of recommended next steps for the claims handler\n\n"
            "Respond with the JSON object only. No explanation."
        )
        raw = self._call_ollama(prompt)
        if not raw:
            return {
                "final_decision": "INVESTIGATE",
                "confidence_score": 0.0,
                "justification": "Unable to generate recommendation – Ollama unavailable.",
                "next_steps": ["Manual review required"],
                "error": "Ollama unavailable",
            }
        return self._extract_json(raw)
