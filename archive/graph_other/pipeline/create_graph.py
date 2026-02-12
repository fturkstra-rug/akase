from constants import DOMAINS, HUMAN_VALUES
import pandas as pd
import argparse
import json
from tqdm import tqdm
from collections import defaultdict, Counter
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartForConditionalGeneration, BartTokenizer
from torch.optim import AdamW
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed_data_file", help="File with preprocessed scraping data.", required=True, type=str)
    parser.add_argument("-v", "--values_file", help="File with predicted human values", required=False, type=str)
    parser.add_argument("-d", "--domain", help="Extract all nodes related to this domain.", required=False, type=str, choices=DOMAINS)
    parser.add_argument("-m", "--model", help="Model to train and used to predict", required=True, type=str)
    return parser.parse_args()

def fit_template_under_512(tokenizer, values, arguments, attainment_type, arg_type):
    """
    Fills in the template with values and arguments, ensuring the total token length stays under 512 tokens.
    Truncates the number of arguments if necessary.
    """
    # Prepare static parts
    val_relation = "attained" if attainment_type == "attain" else "constrained"
    arg_relation = "support" if arg_type == "pro" else "attack"
    prefix = f"Generate Issue: Values: {values} --> are {val_relation} by --> Arguments: "
    suffix = f" --> {arg_relation} --> Issue:"

    # Compute token budget
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    max_total_tokens = 512
    available_tokens = max_total_tokens - len(prefix_tokens) - len(suffix_tokens)

    # Add arguments until we run out of space
    included_args = []
    current_token_count = 0

    for arg in arguments:
        arg_tokens = tokenizer.encode(arg, add_special_tokens=False)
        if current_token_count + len(arg_tokens) > available_tokens:
            break
        included_args.append(arg)
        current_token_count += len(arg_tokens)

    arguments_str = " ||| ".join(included_args)
    final_text = prefix + arguments_str + suffix

    return final_text, included_args


def main():
    args = create_arg_parser()

    # Load data
    with open(args.seed_data_file, 'r') as f:
        seed_data = json.load(f)

    subgraph = args.domain is not None
    values_df = pd.read_csv(args.values_file, sep='\t')

    argument_texts = []
    issue_texts = []
    value_texts = []

    # Edges from argument to issue with labels
    support_edges = []   # label = 1
    attack_edges = []    # label = 2

    # argument --> human_value
    arg_attains_edges = []
    arg_constrains_edges = []

    # issue --> human_value
    issue_attains_edges = []
    issue_constrains_edges = []

    # Map uuids to indices
    uuid_to_issue_idx = {}
    uuid_to_arg_idx = {}
    arg_to_idx = {}

    print("Constructing graph...")
    for row in tqdm(seed_data):

        # Extract only issues related to domain
        if subgraph:
            domain = row.get("domain", [])
            if not (args.domain in domain):
                continue
        
        # ********************************************************************************
        # Create Issue Nodes
        # ********************************************************************************
        issue_text = row.get("issue", "")
        issue_uuid = row.get("uuid", "")
        if not issue_text or not issue_uuid:
            continue

        issue_idx = len(issue_texts)
        uuid_to_issue_idx[issue_uuid] = issue_idx
        issue_texts.append(issue_text)

        # ********************************************************************************
        # Create Argument Nodes / Argument --> Issue Edges
        # ********************************************************************************

        arguments = row.get("arguments", {})
        if not arguments:
            continue

        for stance in ["pro", "con"]:
            stance_arguments = arguments.get(f"{stance}_arguments", [])
            for i, arg in enumerate(stance_arguments):

                # If the argument does not already exist
                if not (arg in arg_to_idx):
                    arg_idx = len(argument_texts)
                    arg_to_idx[arg] = arg_idx
                    argument_texts.append(arg)
                
                # Add edges
                arg_idx = arg_to_idx[arg]
                if stance == "pro":
                    support_edges.append((arg_idx, issue_idx))
                else:
                    attack_edges.append((arg_idx, issue_idx))
                
                # Map idx to uuid
                arg_uuid = f"{issue_uuid}-{stance}-{i + 1}"
                uuid_to_arg_idx[arg_uuid] = arg_idx

    # ********************************************************************************
    # Create Value Nodes / Issue --> Value & Argument --> Value Edges
    # ********************************************************************************

    # Collapse all sentence predictions into one.
    # Text-ID,  Sentence-ID,    Val1    Val2    Val3
    # example   1               0       1       0
    # example   2               0       0       0
    # example   3               1       1       0

    # Becomes

    # Text-ID,  Sentence-ID,    Val1    Val2    Val3
    # example   1               1       1       0
    value_columns = [f"{value} {attainment}" for attainment in ["constrained", "attained"] for value in HUMAN_VALUES]
    grouped = values_df.groupby("Text-ID")[value_columns].max()

    # print(grouped.sum(axis=1).value_counts())

    def add_arg_value_edge(node_idx, value_idx, attainment_type):
        if attainment_type == "attained":
            arg_attains_edges.append((node_idx, value_idx))
        else:
            arg_constrains_edges.append((node_idx, value_idx))

    for value in HUMAN_VALUES:
        value_idx = len(value_texts)
        value_texts.append(value)

        for attainment_type in ["attained", "constrained"]:
            column_name = value + " " + attainment_type
            value_df = grouped[grouped[column_name] == 1]

            if value_df.empty:
                continue

            # Find the nodes that belong to these rows
            for uuid in value_df.index:

                # If it is an issue uuid
                if max(uuid.rfind("-pro"), uuid.rfind("-con")) < 0:
                    issue_idx = uuid_to_issue_idx[uuid] 
                    if attainment_type == "attained":
                        issue_attains_edges.append((issue_idx, value_idx))
                    else:
                        issue_constrains_edges.append((issue_idx, value_idx))
                # If is an argument uuid
                else:
                    arg_idx = uuid_to_arg_idx[uuid]
                    if attainment_type == "attained":
                        arg_attains_edges.append((arg_idx, value_idx))
                    else:
                        arg_constrains_edges.append((arg_idx, value_idx))

    issue_to_pro_arguments = defaultdict(list)
    for pair in support_edges:
        issue_to_pro_arguments[pair[1]].append(pair[0])

    issue_to_con_arguments = defaultdict(list)
    for pair in attack_edges:
        issue_to_con_arguments[pair[1]].append(pair[0])

    argument_to_attained_values = defaultdict(list)
    for pair in arg_attains_edges:
        argument_to_attained_values[pair[0]] = value_texts[pair[1]]

    argument_to_constrained_values = defaultdict(list)
    for pair in arg_constrains_edges:
        argument_to_constrained_values[pair[0]] = value_texts[pair[1]]

    model_inputs = []
    targets = []
    inverted_inputs = []
    values = []

    if args.model == "t5-base":
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
    elif args.model == "bart-base":
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    for i, issue in tqdm(enumerate(issue_texts)):
        pro_arguments = issue_to_pro_arguments[i]
        con_arguments = issue_to_con_arguments[i]

        pro_attained_values = []
        pro_constrained_values = []
        for arg in pro_arguments:
            attained_value = argument_to_attained_values[arg]
            if isinstance(attained_value, str):
                pro_attained_values.append(attained_value)
            elif isinstance(attained_value, list) and attained_value:
                pro_attained_values.extend(attained_value)

            constrained_value = argument_to_constrained_values[arg]
            if isinstance(constrained_value, str):
                pro_constrained_values.append(constrained_value)
            elif isinstance(constrained_value, list) and constrained_value:
                pro_constrained_values.extend(constrained_value)

        con_attained_values = []
        con_constrained_values = []
        for arg in con_arguments:
            attained_value = argument_to_attained_values[arg]
            if isinstance(attained_value, str):
                con_attained_values.append(attained_value)
            elif isinstance(attained_value, list) and attained_value:
                con_attained_values.extend(attained_value)

            constrained_value = argument_to_constrained_values[arg]
            if isinstance(constrained_value, str):
                con_constrained_values.append(constrained_value)
            elif isinstance(constrained_value, list) and constrained_value:
                con_constrained_values.extend(constrained_value)

        pro_attained_values = ', '.join(set(pro_attained_values))
        pro_constrained_values = ', '.join(set(pro_constrained_values))
        con_attained_values = ', '.join(set(con_attained_values))
        con_constrained_values = ', '.join(set(con_constrained_values))

        if pro_arguments:
            arguments = [argument_texts[arg] for arg in pro_arguments]
            if pro_attained_values:
                model_input, included_args = fit_template_under_512(tokenizer, pro_attained_values, arguments, "attain", "pro")
                model_inputs.append(model_input)
                included_values = []
                for arg in included_args:
                    arg_index = argument_texts.index(arg)
                    if x := argument_to_attained_values[arg_index]:
                        included_values.append(x)
                    if y := argument_to_constrained_values[arg_index]:
                        included_values.append(y)
                values.append(included_values)
                inverted_input, included_args = fit_template_under_512(tokenizer, pro_attained_values, arguments, "attain", "con")
                inverted_inputs.append(inverted_input)
                targets.append(issue)

            if pro_constrained_values:
                model_input, included_args = fit_template_under_512(tokenizer, pro_constrained_values, arguments, "constrain", "pro")
                model_inputs.append(model_input)
                included_values = []
                for arg in included_args:
                    arg_index = argument_texts.index(arg)
                    if x := argument_to_attained_values[arg_index]:
                        included_values.append(x)
                    if y := argument_to_constrained_values[arg_index]:
                        included_values.append(y)
                values.append(included_values)
                inverted_input, included_args = fit_template_under_512(tokenizer, pro_constrained_values, arguments, "constrain", "con")
                inverted_inputs.append(inverted_input)
                targets.append(issue)
        
        if con_arguments:
            arguments = [argument_texts[arg] for arg in con_arguments]
            if con_attained_values:
                model_input, included_args = fit_template_under_512(tokenizer, con_attained_values, arguments, "attain", "con")
                model_inputs.append(model_input)
                included_values = []
                for arg in included_args:
                    arg_index = argument_texts.index(arg)
                    if x := argument_to_attained_values[arg_index]:
                        included_values.append(x)
                    if y := argument_to_constrained_values[arg_index]:
                        included_values.append(y)
                values.append(included_values)
                inverted_input, included_args = fit_template_under_512(tokenizer, con_attained_values, arguments, "attain", "pro")
                inverted_inputs.append(inverted_input)
                targets.append(issue)

            if con_constrained_values:
                model_input, included_args = fit_template_under_512(tokenizer, con_constrained_values, arguments, "constrain", "con")
                model_inputs.append(model_input)
                included_values = []
                for arg in included_args:
                    arg_index = argument_texts.index(arg)
                    if x := argument_to_attained_values[arg_index]:
                        included_values.append(x)
                    if y := argument_to_constrained_values[arg_index]:
                        included_values.append(y)
                values.append(included_values)
                inverted_input, included_args = fit_template_under_512(tokenizer, con_constrained_values, arguments, "constrain", "pro")
                inverted_inputs.append(inverted_input)
                targets.append(issue)


    # --------------------------------------------------------------------------------
    # SPLIT TRAIN - TEST (stratify on values)
    # --------------------------------------------------------------------------------

    # Binarize your multi-label outputs
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(values)  # shape: (n_samples, n_labels)

    # Convert other data to numpy arrays (to index more easily)
    model_inputs = np.array(model_inputs)
    targets = np.array(targets)
    inverted_inputs = np.array(inverted_inputs)

    # Set up the stratified splitter
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, test_idx = next(msss.split(model_inputs, Y))

    # Perform the split
    model_inputs_train = model_inputs[train_idx].tolist()
    model_inputs_test = model_inputs[test_idx].tolist()

    targets_train = targets[train_idx].tolist()
    targets_test = targets[test_idx].tolist()

    inverted_inputs_train = inverted_inputs[train_idx].tolist()
    inverted_inputs_test = inverted_inputs[test_idx].tolist()

    values_train = [values[i] for i in train_idx]
    values_test = [values[i] for i in test_idx]

    test_data = {
        "model_inputs": model_inputs_test,
        "inverted_inputs": inverted_inputs_test
    }
    with open('test_data.json', 'w') as f:
        json.dump(test_data, f, indent=4)

    exit()

    def label_distribution(Y, labels):
        return pd.Series(Y.sum(axis=0), index=labels)

    print("Train label distribution:")
    print(label_distribution(mlb.transform(values_train), mlb.classes_))

    print("\nTest label distribution:")
    print(label_distribution(mlb.transform(values_test), mlb.classes_))

    with open('labels_test_set.txt', 'w') as f:
        for target in targets_test:
            f.write(target + '\n')

    # --------------------------------------------------------------------------------
    # T5 finetuning
    # --------------------------------------------------------------------------------

    # Assume these lists are already filled in with correctly formatted strings.    
    # model_inputs = []
    # inverted_inputs = []
    # targets = []

    # Make sure to check the length of each input (max length is 512).

    if args.model == "t5-base":
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
    elif args.model == "bart-base":
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", trust_remote_code=True, use_safetensors=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Dataset class
    class IssueDataset(Dataset):
        def __init__(self, inputs, targets, tokenizer, max_length=512):
            self.inputs = inputs
            self.targets = targets
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            input_text = self.inputs[idx]
            target_text = self.targets[idx]

            input_enc = self.tokenizer(
                input_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            target_enc = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # Flatten tensors (remove batch dimension)
            input_ids = input_enc.input_ids.squeeze()
            attention_mask = input_enc.attention_mask.squeeze()
            labels = target_enc.input_ids.squeeze()

            # Important: replace pad token id's in labels by -100 to ignore in loss
            labels[labels == tokenizer.pad_token_id] = -100

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

    # Hyperparameters
    batch_size = 8
    epochs = 5
    learning_rate = 5e-5
    max_length = 512

    # Prepare dataset and dataloader
    train_dataset = IssueDataset(model_inputs_train, targets_train, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

    # After training, switch model to eval mode
    model.eval()

    # Generate predictions on test set
    results = []
    for input_text in tqdm(model_inputs_test, desc="Generating issues for test set"):
        encoded = tokenizer(input_text, return_tensors='pt', max_length=max_length, truncation=True).to(device)
        generated_ids = model.generate(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        results.append(generated_text)

    # Save outputs to file
    output_path = f"results_{args.model}_test_set.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for issue in results:
            f.write(issue + "\n")

    print(f"Generation complete. Results saved to {output_path}")

    # Generate predictions on inverted inputs
    results = []
    for input_text in tqdm(inverted_inputs_test, desc="Generating issues for inverted test set"):
        encoded = tokenizer(input_text, return_tensors='pt', max_length=max_length, truncation=True).to(device)
        generated_ids = model.generate(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        results.append(generated_text)

    # Save outputs to file
    output_path = f"results_{args.model}_inverted_test_set.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for issue in results:
            f.write(issue + "\n")

    print(f"Generation complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
