import pandas as pd
import re
from pathlib import Path

SPLIT = "test"

previous_model_output = Path(f"deliberation/{SPLIT}_finale_medium/model_outputs.jsonl")

df = pd.read_json(previous_model_output, lines=True)

# Remove the example deliberation tag from the prompt
df['cleaned'] = df['modelInput'].apply(
    lambda x: x['prompt'].replace('<deliberation>', '', 1)
)

# Extract the deliberation so far
df['old'] = df['cleaned'].apply(
    lambda x: re.search(r'<deliberation>(.*?)</deliberation>', x, re.DOTALL).group(1)
)


# Prepare the previous model output
df["new"] = df['modelOutput'].apply(lambda x: f"<deliberator_1>{x['generation'].strip()}</deliberator_1>")

# Remove the vote and confidence tags so the other model doesn't see them
# def clean_tags(text):
#     cleaned = re.sub(r'<vote>.*?</vote>', '', text)
#     cleaned2 = re.sub(r'<confidence>.*?</confidence>', '', cleaned)
#     return cleaned2

# df["new"] = df["new"].apply(clean_tags)

# Add the previous model output to the deliberation
df['argument'] = df['old'] + df['new']

# Remove temporary and unnecessary columns
df.drop(columns=['modelInput', 'modelOutput', 'cleaned', 'old', 'new'], inplace=True)

df.to_json(f"after_medium.jsonl", orient="records", lines=True)


