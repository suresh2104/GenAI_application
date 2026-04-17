import json

def run_llava_inference(prompts, processor, model):
    results = []
    for prompt_data in prompts:
        inputs = processor(
            text=prompt_data["prompt"],
            images=prompt_data["image"],
            return_tensors="pt"
        )
        output = model.generate(**inputs, max_new_tokens=300)
        decoded = processor.decode(output[0][2:], skip_special_tokens=True)
        try:
            json_str = decoded.split('{', 1)[1].rsplit('}', 1)[0]
            results.append(json.loads('{' + json_str + '}'))
        except:
            results.append({"error": "Failed to parse response"})
    return results