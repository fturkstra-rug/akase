import pandas as pd
import re
import random
from sklearn.metrics import classification_report

random.seed(42)

SPLIT = "test"

df1 = pd.read_json(f"deliberation/{SPLIT}_finale_small/model_outputs.jsonl", lines=True)
df2 = pd.read_json(f"deliberation/{SPLIT}_finale_medium/model_outputs.jsonl", lines=True)

def extract_vote(text):
    vote = re.search(r'<vote>(.*?)</vote>', text, re.DOTALL)
    return vote.group(1) if vote else 'x'

def extract_confidence(text):
    confidence = re.search(r'<confidence>(.*?)</confidence>', text, re.DOTALL)
    return confidence.group(1) if confidence else 'x'

df1['small_prediction'] = df1['modelOutput'].apply(lambda x: extract_vote(x['generation'].strip()))
df2['medium_prediction'] = df2['modelOutput'].apply(lambda x: extract_vote(x['generation'].strip()))

# df1['confidence'] = df1['modelOutput'].apply(lambda x: extract_confidence(x['generation'].strip()))
# df2['confidence'] = df2['modelOutput'].apply(lambda x: extract_confidence(x['generation'].strip()))

pred_df = df1.merge(df2[['recordId', 'medium_prediction']], on='recordId', how='left')

# Combine predictions

def choose_prediction(small, medium):
    valid = {'0', '1'}
    if small == medium and small in valid:
        return small
    if small in valid and medium not in valid:
        return small
    if medium in valid and small not in valid:
        return medium
    if small in valid and medium in valid and small != medium:
        return random.choice([small, medium])
    return 'x'

pred_df['prediction'] = pred_df.apply(
    lambda row: choose_prediction(row['small_prediction'], row['medium_prediction']),
    axis=1
)

gold_df = pd.read_json(f'datasets/disagreements_{SPLIT}.jsonl', lines=True)
combined_df = pred_df.merge(gold_df[['recordId', 'label']], on='recordId', how='left')

combined_df.to_json(f"deliberation/{SPLIT}_finale_conclusion/deliberation_predictions.jsonl", orient="records", lines=True)

report = classification_report(combined_df['label'].astype(str), combined_df['prediction'].astype(str), output_dict=True, zero_division=0)
print(report['macro avg']['f1-score'])

def classify_case(small, medium):
    valid = {'0', '1'}
    if small == medium and small in valid:
        return 'agree_valid'
    if small in valid and medium not in valid:
        return 'medium_invalid'
    if medium in valid and small not in valid:
        return 'small_invalid'
    if small in valid and medium in valid and small != medium:
        return 'disagree_valid'
    return 'both_invalid'

pred_df['case'] = pred_df.apply(
    lambda row: classify_case(row['small_prediction'], row['medium_prediction']),
    axis=1
)

case_counts = pred_df['case'].value_counts()
print(case_counts)

disagree_ids = pred_df.loc[pred_df['case'] == 'disagree_valid', 'recordId']
print(disagree_ids)
