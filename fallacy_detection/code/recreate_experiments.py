import pandas as pd

experiments = [
    "zeroshot_llama_small",
    "zeroshot_llama_medium",
    "zeroshot_mistral_small",
    "zeroshot_mistral_medium",
    "zeroshot_cohere_small",
    "zeroshot_cohere_medium",
    "fewshot_llama_small",
    "fewshot_llama_medium",
    "fewshot_mistral_small",
    "fewshot_mistral_medium",
    "fewshot_cohere_small",
    "fewshot_cohere_medium",
    "cot_llama_small",
    "cot_llama_medium",
    "cot_mistral_small",
    "cot_mistral_medium",
    "cot_cohere_small-2",
    "cot_cohere_medium-2"
]

import pandas as pd

val_df = pd.read_json('datasets/val.jsonl', lines=True)
test_df = pd.read_json('datasets/test.jsonl', lines=True)

val_ids = val_df['recordId']
test_ids = test_df['recordId']

for experiment in experiments:
    df = pd.read_json(f'experiments/{experiment}/model_outputs.jsonl', lines=True)

    df_val_outputs = df[df['recordId'].isin(val_ids)]
    df_test_outputs = df[df['recordId'].isin(test_ids)]

    df_val_outputs.to_json(f'experiments/{experiment}/model_outputs_val.jsonl', lines=True, orient='records')
    df_test_outputs.to_json(f'experiments/{experiment}/model_outputs_test.jsonl', lines=True, orient='records')

