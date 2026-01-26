import pandas as pd
from MetaModel import MetaModel
import random
from sklearn.metrics import classification_report

random.seed(42)

SPLIT = "test"

df1 = pd.read_json(f'experiments/zeroshot_llama_medium/model_outputs_{SPLIT}.jsonl', lines=True)
df2 = pd.read_json(f'experiments/fewshot_llama_medium/model_outputs_{SPLIT}.jsonl', lines=True)

df1['zeroshot_prediction'] = MetaModel.extract_output(df1, "medium")
df2['fewshot_prediction'] = MetaModel.extract_output(df2, "medium")

pred_df = df1.merge(df2[['recordId', 'fewshot_prediction']], on='recordId', how='left')

# Combine predictions

def choose_prediction(zero, few):
    valid = {'0', '1'}
    if zero == few and zero in valid:
        return zero
    if zero in valid and few not in valid:
        return zero
    if few in valid and zero not in valid:
        return few
    if zero in valid and few in valid and zero != few:
        # return few
        return random.choice([zero, few])
    return 'x'

pred_df['prediction'] = pred_df.apply(
    lambda row: choose_prediction(row['zeroshot_prediction'], row['fewshot_prediction']),
    axis=1
)

gold_df = pd.read_json(f'datasets/{SPLIT}.jsonl', lines=True)
combined_df = pred_df.merge(gold_df[['recordId', 'label']], on='recordId', how='left')
combined_df.to_json(f"deliberation/{SPLIT}_finale_conclusion/majority_predictions.jsonl", orient="records", lines=True)


report = classification_report(combined_df['label'].astype(str), combined_df['prediction'].astype(str), output_dict=True, zero_division=0)
print(report['accuracy'])

def classify_case(zero, few):
    valid = {'0', '1'}
    if zero == few and zero in valid:
        return 'agree_valid'
    if zero in valid and few not in valid:
        return 'fewshot_invalid'
    if few in valid and zero not in valid:
        return 'zeroshot_invalid'
    if zero in valid and few in valid and zero != few:
        return 'disagree_valid'
    return 'both_invalid'

pred_df['case'] = pred_df.apply(
    lambda row: classify_case(row['zeroshot_prediction'], row['fewshot_prediction']),
    axis=1
)
case_counts = pred_df['case'].value_counts()
print(case_counts)


record_ids = pred_df.loc[pred_df['case'] == 'disagree_valid', "recordId"]
disagree_gold_df = gold_df[gold_df['recordId'].isin(record_ids)]
disagree_gold_df.to_json(f'datasets/disagreements_{SPLIT}.jsonl', orient='records', lines=True)

agree_df = pred_df.loc[pred_df['case'] == 'agree_valid']
agree_df = agree_df.merge(gold_df[['recordId', 'label']], on='recordId', how='left')

print(agree_df['label'].value_counts())

report = classification_report(agree_df['label'].astype(str), agree_df['prediction'].astype(str), output_dict=True, zero_division=0)
print(report['macro avg']['f1-score'], report['accuracy'])




