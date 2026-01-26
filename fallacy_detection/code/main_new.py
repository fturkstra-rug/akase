import argparse
import boto3
import pandas as pd
import logging
from prompt_management import Prompt
from pathlib import Path
from Model import Model
from MetaModel import MetaModel
from AnthropicModel import AnthropicModel
from MistralModel import MistralModel
from DeepseekModel import DeepseekModel
from CohereModel import CohereModel


MINIMUM_BATCH_SIZE = 100

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, help="File to use for the model input")
    parser.add_argument("--batch", required=False, help="Enable/disable batch mode", action='store_true')
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
        path.mkdir(parents=True)
    except FileExistsError:
        user_input = input(f"{path} already exists which means files may be overwritten. Do you want to proceed? (y/n) ").lower()
        if user_input.startswith('y'):
            path.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    dataset = pd.read_json(args.dataset, lines=True)
    logger.info(f"Loaded dataset: {args.dataset}")

    # Init AWS clients
    session = boto3.Session(region_name=args.region)
    bedrock_client = session.client("bedrock")
    bedrock_agent_client = session.client("bedrock-agent")
    bedrock_runtime_client = session.client("bedrock-runtime")
    s3_client = session.client("s3")
    logger.info("Initialized AWS clients.")

    # Load prompt and model
    prompt = Prompt(bedrock_agent_client, id=args.prompt_id, name=args.prompt_name, version=args.prompt_version)
    logger.info(f"Loaded prompt: {prompt}")

    model = Model(s3_client, bedrock_client, bedrock_runtime_client, None, prompt.model_id)
    logger.info(f"Loaded model: {model}")

    # Check if batch mode is enabled
    batch_mode = args.batch
    if args.batch and len(dataset) < MINIMUM_BATCH_SIZE:
        logger.warning(f"Batch mode is enabled but the number of entries ({len(dataset)}) is less than the minimum batch size ({MINIMUM_BATCH_SIZE}). Exiting programs.")
        exit()
 
    logger.info(f"Batch mode set to: {batch_mode}")

    # Get model inputs
    model_inputs = []
    inputs = dataset["argument"]

    for entry in inputs:
        # entry = f"<introduction>Welcome to the fallacy deliberation task. Your goal is to collaboratively assess whether the following argument contains a logical fallacy. Please examine its reasoning carefully before contributing your analysis. Argument: {entry}</introduction>"
        
        rendered = prompt.render([entry])
        formatted = model.format_prompt(rendered, prompt.system)
        body = model.request_body(formatted)

        model_inputs.append(body)
    
    model_inputs_df = pd.DataFrame({"modelInput": model_inputs, "recordId": dataset["recordId"]})
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
