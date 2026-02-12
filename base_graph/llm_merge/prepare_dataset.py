import pandas as pd
import json
import argparse
from tqdm import tqdm
from typing import Optional


def create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the input file")
    parser.add_argument("-o", "--output_file", type=str, required=False, help="Path to the output .jsonl file", default="model_inputs.jsonl")
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

    # Load data
    df = pd.read_csv(input_file)

    cluster_system_prompt = "Your task is to extract the main issue or general concept from the provided sentences.\nIf multiple sentences describe similar issues, merge them into one unified issue.\nOutput the issues in a clear, concise list with one issue per line. Do NOT output anything else."
    noise_system_prompt = "Your task is to extract the main issue or general concept from the provided sentences.\nOutput the issues in a clear, concise list with one issue per line. Do NOT output anything else."

    # Cluster 6
    cluster_example_1 = "This house believes that Western liberal democracies should ban imports of consumer goods made by child labour.\nThis house believes that advanced countries should ban imports of consumer goods made by child labour\nThis house believes that advanced countries should ban imports of consumer goods made by child labor.\nThis house believes that the EU should impose an embargo on countries where child labour exists"
    cluster_example_1_reply = "Banning imports of consumer goods made by child labour.\nImposing embargos on countries where child labour exists."

    # Cluster 14
    cluster_example_2 = "This house supports the normalization of parasocial relationships\nThis house opposes the practice of entertainment companies fostering parasocial relationships between entertainers and their fanbase.\nThis house regrets the rise of parasocial relationships\nThis house regrets the rise of para-social relationships in streaming and social media platforms e.g. Twitch, Tiktok etc\nThis house supports the normalization of parasocial relationships.\nThat we embrace the normalisation of the parasocial relationship with famous people.\nThis house supports the normalisation of the parasocial relationship.\nThis house embrace the normalisation of the parasocial relationship"
    cluster_example_2_reply = "The normalisation of parasocial relationships.\nThe rise of parasocial relationships in streaming and social media.\nEntertainment companies fostering parasocial relationships."

    cluster_examples = {
        cluster_example_1: cluster_example_1_reply,
        cluster_example_2: cluster_example_2_reply,
    }

    noise_example_1 = "This house would say no more ODA.\nThis House Believes That popularity of gym culture brings more harm than good\nThis house, as China, would abandon all claims on the Senkaku Islands\nThis house, being an Italian judge, would convict Berlusconi of abuse of power, corruption and gross negligence and take away all his personal belongings\nThis House Would tax non-renewable energy to match the cost of the cheapest available renewable energy source"
    noise_example_1_reply = "Ending Official Development Assistance (ODA).\nHarmful effects of gym culture.\nChina's claims on the Senkaku Islands.\nProsecution of Berlusconi for abuse of power, corruption and gross negligence.\nTaxation of non-renewable energy to incentivize renewable energy."
    
    noise_example_2 = "This house opposes the ascension of Naftali Bennet to the Israeli premiership.\nThis house would impose gender quota on religious priests.\nThis house would ban cosmetic surgery\nThis house believes that the best way to protect gay rights in the United States is through federal not state policy.\nThis house opposes narratives that depict repression as a bad coping mechanism"
    noise_example_2_reply = "Naftali Bennett's ascension to Israeli premiership.\nGender quotas on religious priests.\nBanning cosmetic surgery.\nFederal policy for the protection of LGBTQ+ rights.\nNarratives depicting repression as a bad coping mechanism."
    
    noise_examples = {
        noise_example_1: noise_example_1_reply,
        noise_example_2: noise_example_2_reply,
    }

    model_inputs = []
    for cluster_id, cluster_df in tqdm(df.groupby("cluster"), desc="Processing clusters"):

        # These are the noise data points that do not belong to any cluster
        if cluster_id == -1:
            num_noise = len(cluster_df)
            batch_size = 10

            for start in range(0, num_noise, batch_size):
                end = min(start + batch_size, num_noise)
                batch = df.iloc[start:end]

                issues = '\n'.join(batch.issue)
                prompt = format_prompt(issues, noise_system_prompt, noise_examples)

                model_input = get_model_input(prompt)
                model_inputs.append(model_input)
            
            continue

        issues = "\n".join(cluster_df.issue)
        prompt = format_prompt(issues, cluster_system_prompt, cluster_examples)
        model_input = get_model_input(prompt)
        model_inputs.append(model_input)

    # Write results to a .jsonl file
    df = pd.DataFrame(model_inputs)
    df.to_json(output_file, orient='records', lines=True)


if __name__ == "__main__":
    main()
