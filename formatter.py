import json

def format_llava_input(images, text_data=None, policy=None):
    prompts = []
    policy_context = (
        "INSURANCE POLICY CONTEXT:\n" +
        json.dumps(policy, indent=2) + "\n\n" if policy else ""
    )
    base_prompt = (
        "<image>\n"
        "You are an insurance claim processing expert. Analyze the damage shown and:\n"
        "1. Identify damage type and severity (1-10)\n"
        "2. Estimate repair cost range (USD)\n"
        "3. Determine coverage eligibility based on policy\n"
        "4. Recommend claim decision (approve, deny, investigate)\n\n"
        f"{policy_context}"
        "Provide response in JSON format with keys: damage_type, severity, cost_range, coverage_eligible, decision."
    )
    for img in images:
        prompts.append({
            "image": img,
            "prompt": base_prompt
        })
    return prompts