import pandas as pd
import re

df = pd.read_json('deliberation/test_finale_conclusion/deliberation_predictions.jsonl', lines=True)
df = pd.read_json('deliberation/test_finale_medium/model_outputs.jsonl', lines=True)


df['prompt'] = df['modelInput'].apply(
    lambda x: x['prompt'].replace('<deliberation>', '', 1)
)

df['old'] = df['prompt'].apply(
    lambda x: re.search(r'<deliberation>(.*?)</deliberation>', x, re.DOTALL).group(1)
)

df["new"] = df['modelOutput'].apply(lambda x: f"<deliberator_2>{x['generation'].strip()}</deliberator_2>")

df['deliberation'] = df['old'] + df['new']

print(df['deliberation'].iloc[0])
