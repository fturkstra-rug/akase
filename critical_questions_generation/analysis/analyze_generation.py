import argparse
from pathlib import Path
import pandas as pd
import json


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True, help="Path to the model outputs file", type=str)
    return parser.parse_args()


def main():
    args = create_argparser()
    input_file = args.input_file

    model_outputs_file = Path(input_file)
    if not model_outputs_file.exists():
        raise FileNotFoundError(f"Error: failed to open file {input_file}")
    
    with open(model_outputs_file, 'r') as f:
        data = [json.loads(line) for line in f]

    input_token_count = sum(row["modelOutput"]["prompt_token_count"] for row in data)
    output_token_count = sum(row["modelOutput"]["generation_token_count"] for row in data)

    print(f"Input (sum) = {input_token_count}")
    print(f"Input (avg) = {input_token_count / len(data):.2f}")
    print(f"Output (sum) = {output_token_count}")
    print(f"Output (avg) = {output_token_count / len(data):.2f}")

    print(data[0]["modelInput"]["prompt"])


if __name__ == "__main__":
    main()