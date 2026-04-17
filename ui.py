import streamlit as st
from .preprocessing import preprocess_file
from .formatter import format_llava_input
from .inference import run_llava_inference
from .postprocess import process_results
from . import config
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText as AutoModelForVision2Seq

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_llava_model():
    processor = AutoProcessor.from_pretrained(config.HF_MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        config.HF_MODEL_ID,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    ).to(DEVICE)
    return processor, model

def main():
    st.title("Insurance Claim Processing AI")
    st.subheader("Upload damage documentation (images, PDFs, videos)")
    processor, model = load_llava_model()
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
                    analysis_results = run_llava_inference(llava_prompts, processor, model)
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
