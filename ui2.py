import streamlit as st
from .preprocessing import preprocess_file
from .formatter import format_llava_input
from .postprocess import process_results
from . import config
import requests
import json

def load_ollama_model():
    return config.OLLAMA_ENDPOINT, config.OLLAMA_MODEL

def run_ollama_inference(prompts, endpoint, model_name):
    results = []
    for prompt_data in prompts:
        payload = {
            "model": model_name,
            "prompt": prompt_data["prompt"],
            "stream": False
        }
        try:
            response = requests.post(endpoint, json=payload, timeout=60)
        except requests.exceptions.ConnectionError:
            results.append({"error": "Could not connect to Ollama. Is it running?"})
            continue
        except requests.exceptions.Timeout:
            results.append({"error": "Ollama request timed out"})
            continue
        if response.status_code == 200:
            try:
                output = response.json()["response"]
                json_str = output.split('{', 1)[1].rsplit('}', 1)[0]
                results.append(json.loads('{' + json_str + '}'))
            except Exception:
                results.append({"error": "Failed to parse response"})
        else:
            results.append({"error": f"Ollama API error: {response.status_code}"})
    return results

def main():
    st.title("Insurance Claim Processing AI")
    st.subheader("Upload damage documentation (images, PDFs, videos)")
    endpoint, model_name = load_ollama_model()
    policy = {
        "policy_number": "INS-2024-001",
        "coverage_types": ["fire", "water", "collision"],
        "deductible": 500,
        "max_coverage": 10000,
        "exclusions": ["wear_and_tear", "intentional_damage"]
    }
    uploaded_file = st.file_uploader(
        "Upload evidence",
        type=["jpg", "jpeg", "png", "pdf", "mp4", "mov"]
    )
    if uploaded_file:
        try:
            images, text_data = preprocess_file(uploaded_file)
        except ValueError as e:
            st.error(str(e))
            return

        if images:
            st.subheader("Uploaded Content")
            cols = st.columns(min(3, len(images)))
            for idx, img in enumerate(images):
                cols[idx % len(cols)].image(img, caption=f"Page/Frame {idx+1}")
            llava_prompts = format_llava_input(images, text_data, policy)

            if st.button("Process Claim", key="process_claim"):
                with st.spinner("Analyzing damage..."):
                    analysis_results = run_ollama_inference(llava_prompts, endpoint, model_name)
                    st.session_state["final_result"] = process_results(analysis_results)

            if "final_result" in st.session_state:
                final_result = st.session_state["final_result"]
                st.subheader("Claim Analysis")
                st.json(final_result)
                st.info(f"Recommended Decision: **{final_result.get('decision', 'investigate').upper()}**")
                st.write(f"**Justification**: {final_result.get('justification', '')}")
                st.subheader("Adjust Decision")
                override = st.selectbox("Final Decision", ["APPROVE", "DENY", "INVESTIGATE"])
                if st.button("Submit Final Decision", key="submit_decision"):
                    st.success(f"Claim {override} submitted for policy {policy['policy_number']}")
                    del st.session_state["final_result"]
        else:
            st.warning("No visual content found in uploaded file")

if __name__ == "__main__":
    main()
