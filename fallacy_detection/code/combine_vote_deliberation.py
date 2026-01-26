import pandas as pd
from sklearn.metrics import classification_report


SPLIT = "test"

df = pd.read_json(f'deliberation/{SPLIT}_finale_conclusion/majority_predictions.jsonl', lines=True)
df_subset = pd.read_json(f'deliberation/{SPLIT}_finale_conclusion/deliberation_predictions.jsonl', lines=True)
df.set_index('recordId', inplace=True)
df_subset.set_index('recordId', inplace=True)

df.update(df_subset)

df['correct'] = df['prediction'].astype(str) == df['label'].astype(str)
print(df['correct'].mean())

report = classification_report(df['label'].astype(str), df['prediction'].astype(str), output_dict=True, zero_division=0)
print(report['accuracy'])

