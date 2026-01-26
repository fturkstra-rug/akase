import boto3
import pandas as pd
from prompt_management import Prompt

# Did your laptop crash and do you still need to save the results and settings?

session = boto3.Session(region_name="us-west-2")
s3_client = session.client("s3")
bedrock_agent_client = session.client("bedrock-agent")

job_id = "msjic5xc8i4z"
output_file = "experiments/cot_llama_medium/model_outputs.jsonl"
settings_file = "experiments/cot_llama_medium/settings.json"
prompt_name = "FD_cot"
prompt_version = "2"

# Download model outputs
s3_client.download_file(
        "batch.output.bucket",
        f"fallacy_detection/{job_id}/model_inputs.jsonl.out",
        output_file
    )
print("Downloaded results to", output_file)

# Save settings
prompt = Prompt(bedrock_agent_client, name=prompt_name, version=prompt_version)
settings_df = pd.DataFrame({
    "prompt": prompt.prompt,
})
settings_df.to_json(settings_file, orient="records")
print("Saved settings to", settings_file)

