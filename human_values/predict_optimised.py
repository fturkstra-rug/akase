import argparse
import os
import pandas as pd
import torch
import transformers
import csv
from tqdm.auto import tqdm
from collections import Counter
from datasets import Dataset
from custom_models.multi_head import MultiHead_MultiLabel_XL
import warnings
import os

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------

values = [
    "Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism", "Achievement",
    "Power: dominance", "Power: resources", "Face", "Security: personal", "Security: societal",
    "Tradition", "Conformity: rules", "Conformity: interpersonal", "Humility", "Benevolence: caring",
    "Benevolence: dependability", "Universalism: concern", "Universalism: nature", "Universalism: tolerance"
]
labels = sum([[value + " attained", value + " constrained"] for value in values], [])
id2label = {idx: label for idx, label in enumerate(labels)}

lang_dict = {
    'EN': 0, 'EL': 1, 'DE': 2, 'TR': 3, 'FR': 4,
    'BG': 5, 'HE': 6, 'IT': 7, 'NL': 8
}

# ----------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------

def thresholds(data):
    data = torch.sigmoid(torch.tensor(data))
    check = torch.zeros_like(data)
    thresholds_list = [
        0.1, 0.3, 0.25, 0.25, 0.25, 0.25, 0.35, 0.3, 0.35, 0.25,
        0.35, 0.15, 0.2, 0.25, 0.1, 0.2, 0.1, 0.15, 0.2, 0.25,
        0.3, 0.25, 0.3, 0.15, 0.1, 0.15, 0.1, 0.3, 0.3, 0.1,
        0.15, 0.4, 0.15, 0.2, 0.1, 0.1, 0.25, 0.2
    ]
    for i, thresh in enumerate(thresholds_list):
        check[:, i] = (data[:, i] >= thresh).float()
    return check

def write_tsv_dataframe(filepath, dataframe):
    dataframe.to_csv(filepath, encoding='utf-8', sep='\t', index=False, header=True, quoting=csv.QUOTE_NONE)

def parse_args():
    parser = argparse.ArgumentParser("predict.py")
    parser.add_argument('-s', '--sentences-dir', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, required=True)
    return parser.parse_args()

# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    model_path = '/scratch/p317595/human_values/models/SotirisLegkas/multi-head-xlm-xl-tokens-38'
    tokenizer_path = '/scratch/p317595/human_values/tokenizer/SotirisLegkas/multi-head-xlm-xl-tokens-38'

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    model = MultiHead_MultiLabel_XL.from_pretrained(model_path, problem_type="multi_label_classification")
    model.to("cuda")
    model.eval()

    # Load sentence data
    sentence_file = os.path.join(args.sentences_dir, "sentences.tsv")
    df = pd.read_csv(sentence_file, sep="\t").dropna()
    df['language'] = df['Text-ID'].apply(lambda x: lang_dict['EN'])  # Replace with x[:2] if needed
    df['pred_labels'] = [None] * len(df)

    checkpoint_interval = 10000
    checkpoint_path = "checkpoint_predictions.tsv"
    
    # Load checkpoint dataframe
    checkpoint_df = pd.read_csv(checkpoint_path, sep="\t")
    
    value_columns = checkpoint_df.columns[2:] # skip Text-ID and Sentence-ID columns
    
    # Get last index where a prediction was made
    prediction_mask = (checkpoint_df[value_columns] == 1).any(axis=1)
    valid_indices = checkpoint_df[prediction_mask].index
    
    if len(valid_indices) > 0:
        last_pred_row = checkpoint_df.loc[valid_indices[-1]]
        last_text_id = last_pred_row["Text-ID"]
        last_sentence_id = last_pred_row["Sentence-ID"]
    
        # Locate this row in the main dataframe
        last_idx = df[(df["Text-ID"] == last_text_id) & (df["Sentence-ID"] == last_sentence_id)].index
    
        if len(last_idx) > 0:
            start_index = last_idx[0] + 1
        else:
            print("Warning: Last predicted row not found in main dataframe.")
            start_index = 0
    else:
        print("No predictions found in checkpoint file.")
        start_index = 0
            
    print(f"Start index set to: {start_index}")
    
    for i in tqdm(range(start_index, len(df))):
        # Prepare context-aware input
        if i == 0:
            text = df.loc[i, "Text"]
        elif i == 1:
            prev_labels = df.loc[i - 1, "pred_labels"]
            context = df.loc[i - 1, "Text"]
            if prev_labels is not None and any(prev_labels):
                context += ''.join([f' <{id2label[k]}>' for k, v in enumerate(prev_labels) if v])
            else:
                context += ' <NONE>'
            text = context + ' </s> ' + df.loc[i, "Text"]
        else:
            labels_1 = df.loc[i - 1, "pred_labels"]
            labels_2 = df.loc[i - 2, "pred_labels"]
            ctx1 = df.loc[i - 1, "Text"]
            ctx2 = df.loc[i - 2, "Text"]
            if labels_2 is not None and any(labels_2):
                ctx2 += ''.join([f' <{id2label[k]}>' for k, v in enumerate(labels_2) if v])
            else:
                ctx2 += ' <NONE>'
            if labels_1 is not None and any(labels_1):
                ctx1 += ''.join([f' <{id2label[k]}>' for k, v in enumerate(labels_1) if v])
            else:
                ctx1 += ' <NONE>'
            text = ctx2 + ' </s> ' + ctx1 + ' </s> ' + df.loc[i, "Text"]

        # Tokenize and predict
        encoded = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
        encoded = {k: v.to("cuda") for k, v in encoded.items()}
        language_tensor = torch.tensor([df.loc[i, "language"]], device="cuda")
        
        with torch.no_grad():
            logits = model(**encoded, language=language_tensor).logits.cpu()
            pred = thresholds(logits)[0].numpy().tolist()
            df.at[i, "pred_labels"] = pred

        # Save checkpoint every 10,000 steps
        if (i + 1) % checkpoint_interval == 0:
            print(f"Checkpoint reached at index {i + 1}, saving partial results...")
        
            # Replace any None or invalid entries with [0]*len(labels)
            invalid_mask = df["pred_labels"].apply(lambda x: not isinstance(x, list) or len(x) != len(labels))
            num_invalid = invalid_mask.sum()
        
            if num_invalid > 0:
                print(f"[Checkpoint] Found {num_invalid} invalid predictions up to index {i + 1}. Replacing with default zeroed lists.")
                df.loc[invalid_mask, "pred_labels"] = df.loc[invalid_mask, "pred_labels"].apply(lambda _: [0] * len(labels))
        
            result_df = pd.DataFrame(df["pred_labels"].tolist(), columns=labels)
            checkpoint_df = df[["Text-ID", "Sentence-ID"]].copy()
            checkpoint_df = pd.concat([checkpoint_df, result_df], axis=1)
            write_tsv_dataframe(checkpoint_path, checkpoint_df)

    # Replace any None or invalid entries with [0]*len(labels)
    invalid_mask = df["pred_labels"].apply(lambda x: not isinstance(x, list) or len(x) != len(labels))
    num_invalid = invalid_mask.sum()
    
    if num_invalid > 0:
        print(f"Found {num_invalid} invalid or missing predictions. Replacing with default zeroed lists.")
        df.loc[invalid_mask, "pred_labels"] = df.loc[invalid_mask, "pred_labels"].apply(lambda _: [0] * len(labels))

        
    # Save final results
    result_df = pd.DataFrame(df["pred_labels"].tolist(), columns=labels)
    final_df = df[["Text-ID", "Sentence-ID"]].copy()
    final_df = pd.concat([final_df, result_df], axis=1)
    write_tsv_dataframe(args.output_file, final_df)
