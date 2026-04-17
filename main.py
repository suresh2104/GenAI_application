"""
Insurance Claim Automation System – Main Streamlit Application.

Full pipeline:
  Upload Evidence → Accident Report → AI Analysis → Decision & Communication
"""

from __future__ import annotations

import json
import uuid
from datetime import date
from typing import Any, Dict, Optional

import streamlit as st
import torch

import config
from preprocessing import preprocess_file
from formatter import format_llava_input
from postprocess import process_results

# ── conditional imports for model backends ──────────────────────────────────
from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq, AutoProcessor

from rag.policy_store import PolicyStore
from rag.claims_db import ClaimsDatabase
from rag.retriever import RAGRetriever
from text_analysis.analyzer import TextAnalyzer
from communication.email_generator import EmailGenerator

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading vision model…")
def load_hf_model():
    processor = AutoProcessor.from_pretrained(config.HF_MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        config.HF_MODEL_ID,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    ).to(DEVICE)
    return processor, model


@st.cache_resource(show_spinner="Initialising policy store…")
def load_rag_components():
    policy_store = PolicyStore(
        embed_model=config.EMBED_MODEL,
        store_path=config.VECTOR_STORE_PATH,
    )
    claims_db = ClaimsDatabase(db_path=config.CLAIMS_DB_PATH)
    retriever = RAGRetriever(policy_store, claims_db)
    return policy_store, claims_db, retriever


# ---------------------------------------------------------------------------
# Sample policies (displayed in sidebar selector)
# ---------------------------------------------------------------------------
SAMPLE_POLICIES = {
    "INS-2024-001 – John Smith": {
        "policy_number": "INS-2024-001",
        "customer_name": "John Smith",
        "customer_email": "john.smith@email.com",
        "coverage_types": ["fire", "water damage", "collision"],
        "deductible": 500,
        "max_coverage": 10000,
        "exclusions": ["intentional damage", "war"],
    },
    "INS-2024-002 – Jane Doe": {
        "policy_number": "INS-2024-002",
        "customer_name": "Jane Doe",
        "customer_email": "jane.doe@email.com",
        "coverage_types": ["collision", "theft"],
        "deductible": 1000,
        "max_coverage": 25000,
        "exclusions": ["racing", "DUI incidents", "cosmetic damage"],
    },
    "INS-2024-003 – Bob Johnson": {
        "policy_number": "INS-2024-003",
        "customer_name": "Bob Johnson",
        "customer_email": "bob.johnson@email.com",
        "coverage_types": ["fire", "water damage", "collision", "theft", "natural disaster", "vandalism"],
        "deductible": 250,
        "max_coverage": 50000,
        "exclusions": ["mechanical breakdown", "normal wear and tear"],
    },
}

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_session():
    defaults = {
        "step": 1,
        "images": [],
        "text_data": None,
        "accident_report": "",
        "selected_policy": None,
        "vision_results": None,
        "text_analysis": None,
        "policy_context": "",
        "claims_history": "",
        "recommendation": None,
        "email_draft": "",
        "claim_id": None,
        "final_decision": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    st.sidebar.title("⚙️ Configuration")

    backend = st.sidebar.radio(
        "Model Backend",
        ["Ollama (local LLM)", "HuggingFace (SmolVLM)"],
        index=0 if config.MODEL_BACKEND == "ollama" else 1,
    )
    st.session_state["model_backend"] = "ollama" if "Ollama" in backend else "huggingface"

    st.sidebar.markdown("---")
    st.sidebar.subheader("Policy Selection")
    policy_label = st.sidebar.selectbox("Select Policy", list(SAMPLE_POLICIES.keys()))
    st.session_state["selected_policy"] = SAMPLE_POLICIES[policy_label]

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Device: `{DEVICE}`  \n"
        f"Backend: `{st.session_state['model_backend']}`"
    )

    # Progress indicator
    st.sidebar.markdown("---")
    st.sidebar.subheader("Progress")
    steps = ["Upload Evidence", "Accident Report", "AI Analysis", "Decision"]
    for i, s in enumerate(steps, start=1):
        icon = "✅" if st.session_state["step"] > i else ("▶️" if st.session_state["step"] == i else "⬜")
        st.sidebar.write(f"{icon} Step {i}: {s}")


# ---------------------------------------------------------------------------
# Step 1 – Upload Evidence
# ---------------------------------------------------------------------------

def step_upload():
    st.header("Step 1 — Upload Damage Evidence")
    st.write("Upload photographs, PDFs (e.g. repair estimates), or video footage of the damage.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["jpg", "jpeg", "png", "pdf", "mp4", "mov"],
        key="file_uploader",
    )

    if uploaded_file:
        with st.spinner("Processing file…"):
            images, text_data = preprocess_file(uploaded_file)

        if images:
            st.success(f"Loaded {len(images)} image(s) / frame(s).")
            cols = st.columns(min(3, len(images)))
            for idx, img in enumerate(images):
                cols[idx % len(cols)].image(img, caption=f"Frame/Page {idx + 1}", use_column_width=True)

            st.session_state["images"] = images
            st.session_state["text_data"] = text_data

            if st.button("▶ Next: Enter Accident Report", type="primary"):
                st.session_state["step"] = 2
                st.rerun()
        else:
            st.warning("No visual content found. Please upload an image, PDF, or video file.")


# ---------------------------------------------------------------------------
# Step 2 – Accident Report
# ---------------------------------------------------------------------------

def step_accident_report():
    st.header("Step 2 — Accident Report")
    policy = st.session_state["selected_policy"]
    st.info(
        f"**Policy:** {policy['policy_number']}  |  "
        f"**Holder:** {policy['customer_name']}  |  "
        f"**Coverage:** {', '.join(policy['coverage_types'])}"
    )

    report = st.text_area(
        "Describe the incident in detail:",
        value=st.session_state["accident_report"],
        height=200,
        placeholder=(
            "e.g. On March 20th 2026, my vehicle was rear-ended at the intersection of "
            "Main St and 5th Ave. A police report was filed (report #2026-0320-007). "
            "The rear bumper and trunk are severely damaged. No injuries were sustained."
        ),
    )
    st.session_state["accident_report"] = report

    col1, col2 = st.columns(2)
    with col1:
        if st.button("◀ Back", type="secondary"):
            st.session_state["step"] = 1
            st.rerun()
    with col2:
        if st.button("▶ Next: Run AI Analysis", type="primary", disabled=not report.strip()):
            st.session_state["step"] = 3
            st.rerun()


# ---------------------------------------------------------------------------
# Step 3 – AI Analysis
# ---------------------------------------------------------------------------

def step_analysis(policy_store, claims_db, retriever):
    st.header("Step 3 — AI Analysis")

    policy = st.session_state["selected_policy"]
    images = st.session_state["images"]
    accident_report = st.session_state["accident_report"]

    analyzer = TextAnalyzer(
        endpoint=config.OLLAMA_ENDPOINT,
        model_name=config.OLLAMA_MODEL,
    )

    if st.button("🚀 Run Full AI Analysis", type="primary"):
        claim_id = f"CLM-{date.today().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        st.session_state["claim_id"] = claim_id

        # ── RAG retrieval ──────────────────────────────────────────────
        with st.spinner("🔍 Retrieving policy context and claims history…"):
            ctx = retriever.retrieve_context(
                query=accident_report,
                policy_number=policy["policy_number"],
            )
            policy_context = "\n\n".join(ctx["policy_chunks"])
            claims_history = ctx["claims_history"]
            st.session_state["policy_context"] = policy_context
            st.session_state["claims_history"] = claims_history

        with st.expander("📄 Retrieved Policy Context", expanded=False):
            st.write(policy_context)
        with st.expander("📋 Claims History", expanded=False):
            st.write(claims_history)

        # ── Vision model analysis ──────────────────────────────────────
        with st.spinner("🔬 Running visual damage assessment…"):
            llava_prompts = format_llava_input(images, st.session_state["text_data"], policy)

            if st.session_state.get("model_backend") == "huggingface":
                from inference import run_llava_inference
                processor, model = load_hf_model()
                vision_raw = run_llava_inference(llava_prompts, processor, model)
            else:
                import requests as req
                vision_raw = []
                for pd in llava_prompts:
                    payload = {
                        "model": config.OLLAMA_MODEL,
                        "prompt": pd["prompt"],
                        "stream": False,
                    }
                    try:
                        r = req.post(config.OLLAMA_ENDPOINT, json=payload, timeout=60)
                        output = r.json().get("response", "")
                        import re, json as _json
                        output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)
                        m = re.search(r"\{.*\}", output, re.DOTALL)
                        vision_raw.append(_json.loads(m.group()) if m else {"error": "parse error"})
                    except Exception as e:
                        vision_raw.append({"error": str(e)})

            vision_result = process_results(vision_raw)
            st.session_state["vision_results"] = vision_result

        st.subheader("🔬 Visual Damage Assessment")
        col1, col2, col3 = st.columns(3)
        col1.metric("Damage Type", vision_result.get("damage_type", "—"))
        col2.metric("Severity", str(vision_result.get("severity", "—")))
        col3.metric("Cost Estimate", vision_result.get("cost_range", "—"))
        st.json(vision_result)

        # ── Text analysis ──────────────────────────────────────────────
        with st.spinner("📝 Analysing accident report…"):
            text_result = analyzer.analyze_accident_report(accident_report, policy_context)
            st.session_state["text_analysis"] = text_result

        st.subheader("📝 Accident Report Analysis")
        st.json(text_result)

        # ── Policy cross-check ─────────────────────────────────────────
        with st.spinner("⚖️ Cross-checking with policy terms…"):
            cross_check = analyzer.cross_check_policy(vision_result, policy_context)

        st.subheader("⚖️ Policy Cross-Check")
        st.json(cross_check)

        # ── Final recommendation ───────────────────────────────────────
        with st.spinner("🧠 Generating final recommendation…"):
            recommendation = analyzer.generate_recommendation(
                vision_result, text_result, claims_history
            )
            st.session_state["recommendation"] = recommendation

        st.subheader("🧠 AI Recommendation")
        decision = recommendation.get("final_decision", "INVESTIGATE")
        colour = {"APPROVE": "green", "DENY": "red", "INVESTIGATE": "orange"}.get(decision, "orange")
        st.markdown(f"### :{colour}[{decision}]")
        st.write(recommendation.get("justification", ""))
        st.json(recommendation)

        if st.button("▶ Next: Review & Communicate", type="primary"):
            st.session_state["step"] = 4
            st.rerun()

    col1, _ = st.columns(2)
    with col1:
        if st.button("◀ Back", type="secondary"):
            st.session_state["step"] = 2
            st.rerun()


# ---------------------------------------------------------------------------
# Step 4 – Decision & Communication
# ---------------------------------------------------------------------------

def step_decision(claims_db):
    st.header("Step 4 — Decision & Customer Communication")

    policy = st.session_state["selected_policy"]
    recommendation = st.session_state.get("recommendation") or {}
    claim_id = st.session_state.get("claim_id", "N/A")
    ai_decision = recommendation.get("final_decision", "INVESTIGATE")

    st.subheader("Final Decision")
    st.info(f"AI Recommendation: **{ai_decision}**")

    final_decision = st.selectbox(
        "Override / Confirm Decision:",
        ["APPROVE", "DENY", "INVESTIGATE"],
        index=["APPROVE", "DENY", "INVESTIGATE"].index(ai_decision)
        if ai_decision in ["APPROVE", "DENY", "INVESTIGATE"]
        else 2,
    )
    st.session_state["final_decision"] = final_decision

    # ── Email generation ───────────────────────────────────────────────
    st.subheader("📧 Customer Email")

    if st.button("Generate Email Draft"):
        email_gen = EmailGenerator(
            endpoint=config.OLLAMA_ENDPOINT,
            model_name=config.OLLAMA_MODEL,
        )
        claim_report = {
            "claim_id": claim_id,
            "customer_name": policy["customer_name"],
            "customer_email": policy["customer_email"],
            "policy_number": policy["policy_number"],
            "recommendation": final_decision,
            "justification": recommendation.get("justification", ""),
            "next_steps": recommendation.get("next_steps", []),
            "damage_type": (st.session_state.get("vision_results") or {}).get("damage_type", ""),
            "cost_range": (st.session_state.get("vision_results") or {}).get("cost_range", ""),
        }
        with st.spinner("Drafting email…"):
            email_draft = email_gen.generate_claim_email(claim_report)
        st.session_state["email_draft"] = email_draft

    if st.session_state.get("email_draft"):
        st.text_area("Email Draft (editable):", value=st.session_state["email_draft"], height=300)

    # ── Submit ─────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("✅ Submit Final Decision", type="primary"):
        vision = st.session_state.get("vision_results") or {}
        record = {
            "claim_id": claim_id,
            "policy_number": policy["policy_number"],
            "claim_date": date.today().isoformat(),
            "damage_type": vision.get("damage_type", "unknown"),
            "severity": str(vision.get("severity", "unknown")),
            "cost_estimate": float(
                str(vision.get("cost_range", "0"))
                .replace("$", "").replace(",", "").split("-")[0].strip() or 0
            ),
            "decision": final_decision,
            "status": "CLOSED" if final_decision in ("APPROVE", "DENY") else "UNDER_REVIEW",
        }
        try:
            saved_id = claims_db.add_claim(record)
            st.success(f"✅ Claim **{saved_id}** submitted. Decision: **{final_decision}**")
            st.balloons()
        except Exception as e:
            st.error(f"Failed to save claim: {e}")

    if st.button("◀ Back", type="secondary"):
        st.session_state["step"] = 3
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    init_session()
    render_sidebar()

    st.title(f"🏥 {config.APP_TITLE}")
    st.markdown(
        "AI-powered end-to-end insurance claim processing: "
        "vision analysis · RAG policy retrieval · LLM decision support · automated communication."
    )
    st.markdown("---")

    policy_store, claims_db, retriever = load_rag_components()

    step = st.session_state["step"]
    if step == 1:
        step_upload()
    elif step == 2:
        step_accident_report()
    elif step == 3:
        step_analysis(policy_store, claims_db, retriever)
    elif step == 4:
        step_decision(claims_db)


if __name__ == "__main__":
    main()
