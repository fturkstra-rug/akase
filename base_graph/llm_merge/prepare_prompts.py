import json
import pandas as pd
from collections import defaultdict


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

# Load prompt
cluster_system_prompt = """Your task is to extract the main issue or general concept from the provided sentences.
If there are sentences that describe similar issues, merge them into one unified issue.
Keep track of the indices of the sentences that you have merged.
You should use all the sentences and refer to the index of each sentence at least once.
Output your answer in the following JSON format:
{"extracted issue": [indices]}
Do NOT output anything else."""

noise_system_prompt = "Your task is to extract the main issue or general concept from the provided sentence.\nDo NOT output anything else."

cluster_examples = {
    """0. This house believes that Western liberal democracies should ban imports of consumer goods made by child labour.
    1. This house believes that advanced countries should ban imports of consumer goods made by child labour
    2. This house believes that advanced countries should ban imports of consumer goods made by child labor.
    3. This house believes that the EU should impose an embargo on countries where child labour exists""": '{"Banning imports of consumer goods made by child labour.": [0,1,2], "Imposing embargos on countries where child labour exists.": [3]}',
    """0. This house supports the normalization of parasocial relationships
    1. This house opposes the practice of entertainment companies fostering parasocial relationships between entertainers and their fanbase.
    2. This house regrets the rise of parasocial relationships
    3. This house regrets the rise of para-social relationships in streaming and social media platforms e.g. Twitch, Tiktok etc
    4. This house supports the normalization of parasocial relationships.
    5. That we embrace the normalisation of the parasocial relationship with famous people.
    6. This house supports the normalisation of the parasocial relationship.
    7. This house embrace the normalisation of the parasocial relationship""": '{"The normalisation of parasocial relationships.": [0,4,5,6,7], "The rise of parasocial relationships in streaming and social media.": [2,3], "Entertainment companies fostering parasocial relationships.": [1]}',
}

noise_examples = {
    "This house would say no more ODA.": "Ending Official Development Assistance (ODA).",
    "This House Believes That popularity of gym culture brings more harm than good": "Harmful effects of gym culture.",
    "This house, as China, would abandon all claims on the Senkaku Islands": "China's claims on the Senkaku Islands.",
    "This house, being an Italian judge, would convict Berlusconi of abuse of power, corruption and gross negligence and take away all his personal belongings": "Prosecution of Berlusconi for abuse of power, corruption and gross negligence.",
    "This House Would tax non-renewable energy to match the cost of the cheapest available renewable energy source": "Taxation of non-renewable energy to incentivize renewable energy.",
    "This house opposes the ascension of Naftali Bennet to the Israeli premiership.": "Naftali Bennett's ascension to Israeli premiership.",
    "This house would impose gender quota on religious priests.": "Gender quotas on religious priests.",
    "This house would ban cosmetic surgery": "Banning cosmetic surgery.",
    "This house believes that the best way to protect gay rights in the United States is through federal not state policy.": "Federal policy for the protection of LGBTQ+ rights.",
    "This house opposes narratives that depict repression as a bad coping mechanism": "Narratives depicting repression as a bad coping mechanism.",
}

# Load data
with open('seed_data_raw.json', 'r') as f:
    data = json.load(f)

# Group data by cluster
clusters = defaultdict(list)
for row in data:
    cluster_id = row['cluster']
    clusters[cluster_id].append(row)

# Prepare model inputs
model_inputs = []
for cluster_id, rows in clusters.items():
    if cluster_id == -1:
        for row in rows:
            prompt = format_prompt(row['motion'], noise_system_prompt, noise_examples)
            model_input = get_model_input(prompt)
            model_inputs.append({"modelInput": model_input, "recordId": row["uuid"]})
    else:   
        # text = '\n'.join([row['motion'] for row in rows])
        text = '\n'.join([f"{i}. {row['motion']}" for i, row in enumerate(rows)])
        prompt = format_prompt(text, cluster_system_prompt, cluster_examples)
        model_input = get_model_input(prompt)
        model_inputs.append({"modelInput": model_input, "recordId": str(cluster_id)})


df = pd.DataFrame(model_inputs)
df.to_json('model_inputs.jsonl', orient='records', lines=True)
