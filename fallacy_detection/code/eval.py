import argparse
import pandas as pd
import json
from datasets import Dataset as HuggingFaceDataset
from Dataset import Dataset
from sklearn.metrics import classification_report
from Model import Model
from MetaModel import MetaModel
from AnthropicModel import AnthropicModel
from MafaldaDataset import MafaldaDataset
from TestDataset import TestDataset
from LogicDataset import LogicDataset
from LogicClimateDataset import LogicClimateDataset
from RuFalDataset import RuFalDataset
from ElecDeb60to20 import ElecDeb60to20Dataset
from CoCoLoFaDataset import CoCoLoFaDataset
from NonFallaciousDataset import NonFallaciousDataset
from MistralModel import MistralModel
from CohereModel import CohereModel
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

def create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="Task on which to evaluate", default="det", choices=["det", "cls"])
    return parser.parse_args()

def main():
    args = create_arg_parser()

    # Extract labels from the datasets
    dataset_names = ["mafalda", "logic", "logicclimate", "rufal", "elecdeb60to20", "cocolofa", "non-fallacious"]
    # dataset_names = ["cocolofa", "non-fallacious"]
    datasets = [Dataset(name) for name in dataset_names]
    
    labels = []
    for dataset in datasets:
        labels.extend(dataset.labels)

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

        if size != 'small-2':
            continue

        model_outputs_df = pd.read_json(experiment / "model_outputs.jsonl", lines=True)
        # model_inputs_df = pd.read_json(experiment / "model_inputs.jsonl", lines=True)
        model_outputs_df['recordId'] = model_outputs_df['recordId'].astype(int)
        model_outputs_df = model_outputs_df.sort_values(by='recordId')
        
        # model_outputs_df = model_outputs_df[model_inputs_df['dataset'].isin(["cocolofa", "non-fallacious"])]
        # df = df.reset_index(drop=True)

        model_class = model_mappings[model]
        predictions = model_class.extract_output(model_outputs_df, size)
        invalid = sum(1 for p in predictions if p not in ('0', '1'))
        print("Invalid", invalid)

        # print(f"{experiment_name}\t\t'0' - {(predictions == '0').sum()}\t '1' - {(predictions == '1').sum()}")

        correct = 0
        for label, pred in zip(labels, predictions):
            if pred in ('0', '1') and pred == label:
                correct += 1

        accuracy = correct / len(labels)

        # print(technique, model, size, accuracy)
        result = f"{experiment_name}\t-\t{accuracy * 100:.1f}"
        results.append(result)

        report = classification_report(labels, predictions, labels=["0", "1"], output_dict=True, zero_division=0)

        with open(f"{experiment}/eval.json", "w") as f:
            json.dump(report, f, indent=2)

    # TODO Evaluate per dataset
    print('\n'.join(sorted(results)))
        

if __name__ == "__main__":
    main()
