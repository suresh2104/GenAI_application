"""
AI-powered email generator for insurance claim status communications.

Uses Ollama to draft personalized, professional emails to policyholders.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


class EmailGenerator:
    """
    Generates personalized claim status emails using an Ollama-hosted LLM.
    Falls back to a structured template when Ollama is unavailable.
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
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        try:
            resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            # Strip <think> blocks from DeepSeek-R1
            return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama not reachable – using template fallback.")
            return ""
        except Exception as exc:
            logger.error("Email generation failed: %s", exc)
            return ""

    def _build_email_prompt(self, claim_report: Dict[str, Any]) -> str:
        return (
            "You are a professional insurance claims correspondent.\n\n"
            "Write a clear, empathetic, and professional email to the policyholder "
            "about the status of their insurance claim. Use the details below.\n\n"
            f"CLAIM DETAILS:\n{json.dumps(claim_report, indent=2, default=str)}\n\n"
            "Requirements:\n"
            "- Address the customer by name\n"
            "- State the claim ID and decision clearly\n"
            "- Explain the justification in simple language\n"
            "- List any next steps the customer needs to take\n"
            "- Use a professional but warm tone\n"
            "- Include a subject line at the top (format: Subject: ...)\n"
            "- Sign off as 'Claims Processing Team'\n\n"
            "Write the email now:"
        )

    def _template_fallback(self, claim_report: Dict[str, Any]) -> str:
        """Return a structured template email when Ollama is unavailable."""
        claim_id = claim_report.get("claim_id", "N/A")
        customer_name = claim_report.get("customer_name", "Valued Customer")
        decision = claim_report.get("recommendation", claim_report.get("final_decision", "UNDER REVIEW"))
        justification = claim_report.get("justification", "Your claim is being processed.")
        next_steps = claim_report.get("next_steps", ["Our team will contact you within 3-5 business days."])

        if isinstance(next_steps, list):
            steps_text = "\n".join(f"  • {s}" for s in next_steps)
        else:
            steps_text = f"  • {next_steps}"

        today = date.today().strftime("%B %d, %Y")

        return f"""Subject: Insurance Claim {claim_id} – Decision: {decision}

Dear {customer_name},

Thank you for submitting your insurance claim. We have completed our review of your case.

CLAIM REFERENCE: {claim_id}
DATE OF REVIEW: {today}
DECISION: {decision}

DETAILS:
{justification}

NEXT STEPS:
{steps_text}

If you have any questions or require further assistance, please do not hesitate to contact our claims team.

Yours sincerely,
Claims Processing Team
Insurance Claim Automation System"""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_claim_email(self, claim_report: Dict[str, Any]) -> str:
        """
        Generate a personalized email for the given *claim_report* dict.

        Returns the full email text (subject line + body).
        Falls back to a structured template if Ollama is unavailable.
        """
        prompt = self._build_email_prompt(claim_report)
        email_text = self._call_ollama(prompt)

        if not email_text:
            logger.info("Using template fallback for email generation.")
            email_text = self._template_fallback(claim_report)

        return self.format_email(
            email_text,
            customer_name=claim_report.get("customer_name", ""),
            claim_id=claim_report.get("claim_id", ""),
        )

    def format_email(self, raw_email: str, customer_name: str, claim_id: str) -> str:
        """
        Post-process the raw email to ensure consistent formatting.
        Adds a separator line and strips excessive whitespace.
        """
        raw_email = raw_email.strip()
        # Ensure subject line exists
        if not raw_email.lower().startswith("subject:"):
            raw_email = f"Subject: Regarding Your Insurance Claim {claim_id}\n\n{raw_email}"
        return raw_email
