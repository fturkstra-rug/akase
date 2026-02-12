import pandas as pd
import argparse
from tqdm import tqdm
from typing import Optional


def create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_file", type=str, required=True, help="Path to the input file"
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        required=False,
        help="Path to the output .jsonl file",
        default="model_inputs.jsonl",
    )
    return parser.parse_args()


def get_model_input(prompt, max_gen_len=1024, temperature=0.5, top_p=0.9):
    model_input = {
        "prompt": prompt,
        "max_gen_len": max_gen_len,
        "temperature": temperature,
        "top_p": top_p,
    }
    return {"modelInput": model_input}


def format_prompt(
    user_message: str,
    system_prompt: Optional[str] = None,
    examples: Optional[dict[str, str]] = None,
) -> str:
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


def main():
    args = create_arg_parser()
    input_file = args.input_file
    output_file = args.output_file

    system_prompt = (
        "Your task is to extract the general issue from the provided motions."
    )
    user_example = (
        "This House Believes That popularity of gym culture brings more harm than good"
    )
    assistant_reply = "Harmful effects of gym culture."
    examples = {
        user_example: assistant_reply,
    }

    df = pd.read_csv(input_file)

    model_inputs = []
    for cluster_id, cluster_df in tqdm(
        df.groupby("cluster"), desc="Processing clusters"
    ):
        motions = "\n".join(cluster_df.motions)
        prompt = format_prompt(motions, system_prompt, examples)
        model_input = get_model_input(prompt)
        model_inputs.append(model_input)

    # Write results to a .jsonl file
    df = pd.DataFrame(model_inputs)
    df.to_json(output_file, orient="records", lines=True)


if __name__ == "__main__":
    main()
