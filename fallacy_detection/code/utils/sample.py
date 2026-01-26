import pandas as pd

# Randomly sample 1061 samples from the cocolofa dev set to make the dataset balanced

df = pd.read_json("datasets/cocolofa_train.json")

all_comments = []
for i, article in df.iterrows():
    all_comments.extend(article.get("comments", []))

df = pd.DataFrame(all_comments)
df = df[df['fallacy'] == 'none']

df_sampled = df.sample(n=1061, random_state=42)
df_sampled.to_json('datasets/non-fallacious.jsonl', orient="records", lines=True)
