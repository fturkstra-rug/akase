from Model import Model
from MetaModel import MetaModel
from prompt_management import Prompt
import boto3
from Dataset import Dataset
from MafaldaDataset import MafaldaDataset
from LogicDataset import LogicDataset
from LogicClimateDataset import LogicClimateDataset
from RuFalDataset import RuFalDataset
from ElecDeb60to20 import ElecDeb60to20Dataset
from NonFallaciousDataset import NonFallaciousDataset
from CoCoLoFaDataset import CoCoLoFaDataset

prompt_id = ""
prompt_version = ""
datasets = ["mafalda", "logic", "logicclimate", "rufal", "elecdeb60to20", "cocolofa", "non-fallacious"]

# Initialize AWS clients
session = boto3.Session(region_name="us-west-2")
bedrock_agent_client = session.client("bedrock-agent")
bedrock_runtime_client = session.client("bedrock-runtime")

# Load prompt and model
prompt = Prompt(bedrock_agent_client, id=prompt_id, name=None, version=prompt_version)
print(f"Loaded prompt: {prompt}")

model = Model(None, None, bedrock_runtime_client, None, prompt.model_id)
print(f"Loaded model: {model}")

# Load the datasets
datasets = {name: Dataset(name) for name in datasets}
print(f"Loaded datasets: {','.join(datasets.keys())}")

# Start deliberation and present the argument
context = []

introduction = ""

context += introduction

# Call the first model
output = model.invoke(prompt_1)
context += output
output = model.invoke(prompt_2)
context += output
output = model.invoke(prompt_3)


