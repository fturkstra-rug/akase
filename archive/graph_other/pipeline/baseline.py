import pandas as pd
import json


def format_prompt(user_message, system_prompt=None, examples=None):
    prompt = "<|begin_of_text|>"

    # Add system instructions
    if system_prompt is not None:
        prompt += f"<|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|>"

    # Add in-context examples
    if examples is not None:
        for example, reply in examples.items():
            prompt += f"<|start_header_id|>user<|end_header_id|>{example}<|eot_id|>"
            prompt += f"<|start_header_id|>assistant<|end_header_id|>{reply}<|eot_id|>"

    # Add user prompt
    prompt += f"<|start_header_id|>user<|end_header_id|>{user_message}<|eot_id|>"

    # Add assistant token to start generation
    prompt += "<|start_header_id|>assistant<|end_header_id|>"

    return prompt

def get_model_input(prompt, max_gen_len=1024, temperature=0.5, top_p=0.9):
    return {
        "prompt": prompt,
        "max_gen_len": max_gen_len,
        "temperature": temperature,
        "top_p": top_p,
    }

# Load data
with open('test_data.json', 'r') as f:
    data = json.load(f)

# Prepare model inputs
for input_type in ["model_inputs", "inverted_inputs"]: 

    model_inputs = []
    for i, sample in enumerate(data[input_type]):
        prompt = format_prompt(sample)
        model_input = get_model_input(prompt)
        model_inputs.append({"modelInput": model_input, "recordId": str(i)})

    df = pd.DataFrame(model_inputs)
    df.to_json(f'{input_type}.jsonl', orient='records', lines=True)
