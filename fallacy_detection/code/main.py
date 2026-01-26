import argparse
import boto3
import pandas as pd
import logging
from prompt_management import Prompt
from datasets import Dataset as HuggingFaceDataset, load_dataset
from Dataset import Dataset
from pathlib import Path
from Model import Model
from MetaModel import MetaModel
from AnthropicModel import AnthropicModel
from MafaldaDataset import MafaldaDataset
from MMArgFallacyDataset import MMArgFallacyDataset
from TestDataset import TestDataset
from LogicDataset import LogicDataset
from LogicClimateDataset import LogicClimateDataset
from RuFalDataset import RuFalDataset
from ElecDeb60to20 import ElecDeb60to20Dataset
from NonFallaciousDataset import NonFallaciousDataset
from CoCoLoFaDataset import CoCoLoFaDataset
from MistralModel import MistralModel
from DeepseekModel import DeepseekModel
from CohereModel import CohereModel
import math


# TODO
# NECESSARY AND URGENT
# get inference parameters from prompt

# NECESSARY BUT NOT URGENT
# add debate/deliberation structure, a shell file and using the modelOutputs as a dataset as input for the next script call?

# NOT NECESSARY BUT NICE
# use converse api when chat type, then you can also pass in the inference parameters from the prompt and model-specific parameters
# delete prompt versions (max = 10)
# add more models with their specific request bodies/formatting
# make config file for all settings
# rewrite and refactor the code
# document the code


MINIMUM_BATCH_SIZE = 100
datasets = ["mafalda", "logic", "logicclimate", "rufal", "elecdeb60to20", "cocolofa", "non-fallacious", "test"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str, required=True, help="Datasets to use for the model input", nargs="+", choices=datasets)
    parser.add_argument("--batch", type=str, required=False, help="Enable/disable batch mode", default="auto", choices=["auto", "disable", "force"])
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--deliberation", required=False, help="Enable deliberation mode", action='store_true')
    
    # Fetch prompt either by name or id
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("-pn", "--prompt_name", type=str, help="Prompt name to use for the model input")
    prompt_group.add_argument("-pi", "--prompt_id", type=str, help="Prompt ID to use for the model input")
    parser.add_argument("-pv", "--prompt_version", required=False, help="Optional version of the prompt", default="latest")

    parser.add_argument("-r", "--region", type=str, required=False, help="AWS region for Bedrock", default="us-west-2")

    return parser.parse_args()


def main():
    # Get CLI arguments
    args = create_arg_parser()

    # Create i/o directories
    folder = "experiments" if not args.deliberation else "deliberation"
    path = Path(folder) / args.experiment_name 
    try:
        ignore_test = args.experiment_name == "test"
        path.mkdir(parents=True, exist_ok=ignore_test)
    except FileExistsError:
        user_input = input(f"{path} already exists which means files may be overwritten. Do you want to proceed? (y/n) ").lower()
        if user_input.startswith('y'):
            path.mkdir(parents=True, exist_ok=True)

    # Load the datasets
    datasets = {name: Dataset(name) for name in args.datasets}
    logger.info(f"Loaded datasets: {','.join(datasets.keys())}")

    # Init AWS clients
    session = boto3.Session(region_name=args.region)
    bedrock_client = session.client("bedrock")
    bedrock_agent_client = session.client("bedrock-agent")
    bedrock_runtime_client = session.client("bedrock-runtime")
    s3_client = session.client("s3")
    sagemaker_runtime_client = session.client("sagemaker-runtime")
    logger.info("Initialized AWS clients.")

    # Load prompt and model
    prompt = Prompt(bedrock_agent_client, id=args.prompt_id, name=args.prompt_name, version=args.prompt_version)
    logger.info(f"Loaded prompt: {prompt}")

    model = Model(s3_client, bedrock_client, bedrock_runtime_client, sagemaker_runtime_client, prompt.model_id)
    logger.info(f"Loaded model: {model}")

    # Check if batch mode is enabled
    num_entries = sum(len(dataset) for dataset in datasets.values())

    match args.batch:
        case "force":
            batch_mode = True
        case "disable":
            batch_mode = False
        case "auto":
            batch_mode = num_entries >= MINIMUM_BATCH_SIZE
 
    logger.info(f"Batch mode set to: {batch_mode}")

    # Get model inputs
    model_inputs = []
    record_ids = []
    dataset_names = []
    current_id = 1

    for dataset in datasets.values():
        inputs = dataset.data[dataset.input_key]

        for entry in inputs:
            # if args.deliberation:
                # entry = f"<introduction>Welcome to the fallacy deliberation task. Your goal is to collaboratively assess whether the following argument contains a logical fallacy. Please examine its reasoning carefully before contributing your analysis. Argument: {entry}</introduction>"
            rendered = prompt.render([entry])

            if model.id == "mistral.mistral-large-2407-v1:0" or model.id == "cohere.command-r-v1:0":
                formatted = rendered # TODO Fix this
            else:
                formatted = model.format_prompt(rendered, prompt.system)

            body = model.request_body(formatted)

            model_inputs.append(body)
            record_ids.append(str(current_id))
            dataset_names.append(dataset.name)
            current_id += 1


    if batch_mode:
        multiple = math.ceil(MINIMUM_BATCH_SIZE / num_entries)
        model_inputs *= multiple
        if multiple > 1:
            logger.warning(f"Input data has been multiplied by {multiple}. Results are automatically deduplicated after inference.")
        
    model_inputs_df = pd.DataFrame({"modelInput": model_inputs, "recordId": record_ids, "dataset": dataset_names})
    model_inputs_df.to_json(path / 'model_inputs.jsonl', orient="records", index=False, lines=True)
    logger.info(f"Successfully prepared {len(model_inputs_df)} model_inputs.")

    # Save model outputs
    model.run_inference(path, model_inputs_df, batch_mode)
    logger.info(f"Inference complete. Results are saved to {path / 'model_outputs.jsonl'}")

    # Save other experimental data/settings
    settings_df = pd.DataFrame({
        "prompt": prompt.prompt,
    })
    settings_df.to_json(path / "settings.json", orient="records")
    logger.info(f"Experimental settings are saved to {path / 'settings.json'}")
    

if __name__ == "__main__":
    main()
