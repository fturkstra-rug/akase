import argparse
import pandas as pd
import json
from sklearn.metrics import classification_report
from MetaModel import MetaModel
from MistralModel import MistralModel
from CohereModel import CohereModel
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

SPLIT = "val"

def main():
    # Extract labels from the datasets
    dataset = pd.read_json(f"datasets/{SPLIT}.jsonl", lines=True)
    dataset = dataset.sort_values(by='recordId')
    labels = dataset['label'].tolist()

    # Extract model outputs from the experiments
    experiments_dir = Path("experiments")
    experiments = [d for d in experiments_dir.iterdir() if d.is_dir()]

    model_mappings = {
        "mistral": MistralModel,
        "cohere": CohereModel,
        "llama": MetaModel,
    }

    results = []

    for experiment in experiments:
        experiment_name = experiment.name
        technique, model, size, *ok = experiment_name.split("_")

        try:
            model_outputs_df = pd.read_json(experiment / f"model_outputs_{SPLIT}.jsonl", lines=True)
        except ValueError:
            print(f"Skipping {experiment_name} due to missing model outputs for split {SPLIT}.")
            continue

        model_outputs_df['recordId'] = model_outputs_df['recordId'].astype(int)
        model_outputs_df = model_outputs_df.sort_values(by='recordId')
        
        model_class = model_mappings[model]
        predictions = model_class.extract_output(model_outputs_df, size)

        correct = 0
        for label, pred in zip(labels, predictions):
            if pred in ('0', '1') and pred == str(label):
                correct += 1

        accuracy = correct / len(labels)

        # print(technique, model, size, accuracy)
        results.append({"experiment_name": experiment_name, "accuracy": f"{accuracy * 100:.1f}"})


    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="accuracy", ascending=False)
    print(results_df)
        

if __name__ == "__main__":
    main()
