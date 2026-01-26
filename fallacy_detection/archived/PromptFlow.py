import re
import boto3
import logging
from botocore.exceptions import ClientError
from typing import Optional, Union


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_flow(
    prompt_name: Optional[str] = None,
    prompt_id: Optional[str] = None,
    prompt_version: Optional[str] = "DRAFT",
    region_name: str = "us-west-2",
) -> None:
    """
    Finds and initializes the flow that is linked to the specified prompt.
    If no such flow exists, it creates a new one.

    Args:
        prompt_name (str): The name of the prompt.
        prompt_id (str): The ID of the prompt.
        prompt_version (str): The version of the prompt.
        region_name (str): The AWS region name.

    Returns:
        ??
    """

    # Get boto3 client
    session = boto3.Session(profile_name="default")
    bedrock_agent_client = session.client("bedrock-agent", region_name=region_name)

    # Get prompt
    prompt = get_prompt(bedrock_agent_client, prompt_name, prompt_id, prompt_version)
    if not prompt:
        return None
    logger.info(f"Prompt found: {prompt}")

    # Find all flows linked to the prompt
    flows = get_flows(bedrock_agent_client, prompt)
    logger.info(f"Flows found: {flows}")

    # If there are no flows, create a new one
    if len(flows) == 0:
        create_flow()

    # If there is only one flow, run it
    elif len(flows) == 1:
        run_flow(flows.pop())

    # If there are multiple flows, ask the user which flow to run
    else:
        logger.info("Multiple flows found. Please select one to run.")
        for i, flow in enumerate(flows):
            logger.info(f"{i}: {flow['name']}")

        selected_flow_index = -1
        while selected_flow_index < 0 or selected_flow_index >= len(flows):
            try:
                selected_flow_index = int(input("Select a flow to run: "))
            except ValueError:
                logger.error("Invalid input. Please enter a valid number.")
                continue
        
        selected_flow = flows[selected_flow_index]
        logger.info(f"Selected flow: {selected_flow['name']}")
        run_flow(selected_flow)

def create_flow():
    pass

def invoke_flow(client, flow_id, input_data):
    """
    Invoke an Amazon Bedrock flow and handle the response stream.

    Args:
        client: Boto3 client for Amazon Bedrock agent runtime.
        flow_id: The ID of the flow to invoke.
        flow_alias_id: The alias ID of the flow.
        input_data: Input data for the flow.

    Returns:
        Dict containing flow status and flow output.
    """

    response = None
    request_params = None

    request_params = {
            "flowIdentifier": flow_id,
            "flowAliasIdentifier": "KXFD0TQOEE",
            "inputs": [input_data],
            "enableTrace": True
        }


    response = client.invoke_flow(**request_params)

    flow_status = ""
    output= ""

    # Process the streaming response
    for event in response['responseStream']:

        # Check if flow is complete.
        if 'flowCompletionEvent' in event:
            flow_status = event['flowCompletionEvent']['completionReason']

        # Save the model output.
        elif 'flowOutputEvent' in event:
            output = event['flowOutputEvent']['content']['document']
            logger.info("Output : %s", output)

        # Log trace events.
        elif 'flowTraceEvent' in event:
            logger.info("Flow trace:  %s", event['flowTraceEvent'])
    
    return {
        "flow_status": flow_status,
        "output": output
    }

def run_flow(flow):
    print(flow.keys())

def find_prompt_id_by_name(client: boto3.Session.client, prompt_name: str) -> Optional[str]:
    """
    Find the prompt ID by its name.

    Args:
        client: The Bedrock agent client.
        prompt_name (str): The name of the prompt to search for.

    Returns:
        str: The ID of the prompt if found, else None.

    """
    paginator = client.get_paginator("list_prompts")
    for page in paginator.paginate():
        for prompt in page.get("promptSummaries", []):
            if prompt.get("name") == prompt_name:
                return prompt.get("id")


def get_prompt(
    client: boto3.Session.client,
    prompt_name: Optional[str] = None,
    prompt_id: Optional[str] = None,
    prompt_version: Optional[str] = "DRAFT",
) -> Optional[dict]:
    """
    Find a stored prompt in Bedrock Prompt Management by name or ID.
    Optionally, specify a version.
    Args:
        bedrock_agent_client: The Bedrock agent client.
        prompt_name (str, optional): The name of the prompt to search for.
        prompt_id (str, optional): The ID of the prompt to search for.
        prompt_version (str or int, optional): The specific version of the prompt to search for.
    Returns:
        dict or None: The prompt metadata (including id and version) if found, else None.
    """
    if not prompt_id:

        prompt_id = find_prompt_id_by_name(client, prompt_name)
        if not prompt_id:
            logger.error(f"Prompt with name '{prompt_name}' not found.")
            return None
        
    try:
        return client.get_prompt(promptIdentifier=prompt_id, promptVersion=prompt_version)
    except Exception as e:
        logger.error(f"Failed to access prompt with identifier '{prompt_id}': {e}")
        return None


def flow_uses_prompt(flow: dict, prompt_id: str, prompt_version: str) -> bool:
    """
    Check if the flow uses the specified prompt.
    Args:
        flow (dict): The flow metadata.
        prompt_id (str): The ID of the prompt.
        prompt_version (str): The version of the prompt.
    Returns:
        bool: True if the flow uses the prompt, else False.
    """
    nodes = flow.get("definition", {}).get("nodes", [])
    
    for node in nodes:
        config = node.get("configuration", {})
        prompt_config = config.get("prompt", {})
        source_config = prompt_config.get("sourceConfiguration", {})

        if "resource" in source_config:
            prompt_arn = source_config["resource"].get("promptArn")

            if prompt_arn:
                arn_id = prompt_arn.split("prompt/")[-1]

                try:
                    arn_id, arn_version = arn_id.split(":")
                except ValueError:
                    arn_version = "DRAFT"
                
                if arn_id == prompt_id:
                    if prompt_version != "DRAFT":
                        return prompt_version == arn_version
                    else:
                        return True
    return False


def get_flows(client: boto3.Session.client, prompt: dict) -> list:
    """
    Find the flows linked to the specified prompt.

    Args:
        client: The Bedrock agent client.
        prompt (dict): The prompt metadata.

    Returns:
        list: A list of all flows that are linked to the prompt.
    """
    prompt_id = prompt.get("id")
    prompt_version = prompt.get("version")
    flows = []
    
    paginator = client.get_paginator("list_flows")
    for page in paginator.paginate():
        for flow in page.get("flowSummaries", []):
            flow_definition = client.get_flow(flowIdentifier=flow["id"])
                
            if flow_uses_prompt(flow_definition, prompt_id, prompt_version):
                flows.append(flow_definition)

    return flows



class FlowManager:
    def __init__(
        self,
        prompt_name=None,
        prompt_id=None,
        prompt_version="DRAFT",
        region_name="us-west-2",
    ):
        """ """
        self.prompt_name = prompt_name
        self.prompt_id = prompt_id
        self.prompt_version = prompt_version
        self.region_name = region_name

        # Get various boto3 clients
        session = boto3.Session(profile_name="default")
        self.bedrock_client = session.client("bedrock", region_name=region_name)
        self.bedrock_agent_client = session.client(
            "bedrock-agent", region_name=region_name
        )
        self.bedrock_agent_runtime_client = session.client(
            "bedrock-agent-runtime", region_name=region_name
        )

        self.list_flows()
        # self.prompt = self.find_stored_prompt(prompt_name, prompt_id, prompt_version)
        # print(self.prompt)

    def extract_prompt_arns(self, flow_definition):
        prompt_arns = []
        for node in flow_definition.get("definition", {}).get("nodes", []):
            config = node.get("configuration", {})
            prompt_config = config.get("prompt", {})
            source_config = prompt_config.get("sourceConfiguration", {})
            if "resource" in source_config:
                prompt_arn = source_config["resource"].get("promptArn")
                if prompt_arn:
                    prompt_arns.append(prompt_arn)
        return prompt_arns

    def list_flows(self):
        try:
            paginator = self.bedrock_agent_client.get_paginator("list_flows")
            for page in paginator.paginate():
                for flow in page.get("flowSummaries", []):
                    flow_definition = self.bedrock_agent_client.get_flow(
                        flowIdentifier=flow["id"]
                    )
                    print(self.extract_prompt_arns(flow_definition))

        except Exception as e:
            print(f"An error occurred: {e}")

    def find_stored_prompt(self, prompt_name: str, prompt_id: str, prompt_version: str):
        """
        Find a stored prompt in Bedrock Prompt Management by name or ID.
        Optionally, specify a version.

        Args:
            prompt_name (str, optional): The name of the prompt to search for.
            prompt_id (str, optional): The ID of the prompt to search for.
            prompt_version (str or int, optional): The specific version of the prompt to search for.

        Returns:
            dict or None: The prompt metadata (including id and version) if found, else None.
        """
        if not prompt_id:

            def find_prompt_id_by_name(paginator):
                for page in paginator.paginate():
                    for prompt in page.get("promptSummaries", []):
                        if prompt.get("name") == prompt_name:
                            return prompt.get("id")

            paginator = self.bedrock_agent_client.get_paginator("list_prompts")
            prompt_id = find_prompt_id_by_name(paginator)

            if not prompt_id:
                print(f"Prompt with name '{prompt_name}' not found.")
                return None

        try:
            return self.bedrock_agent_client.get_prompt(
                promptIdentifier=prompt_id, promptVersion=prompt_version
            )
        except Exception as e:
            print(f"Failed to access prompt with identifier '{prompt_id}': {e}")
            return None

    def create_input_node(self, name: str) -> dict:
        """
        Creates an input node configuration for an Amazon Bedrock flow.

        The input node serves as the entry point for the flow and defines
        the initial document structure that will be passed to subsequent nodes.

        Args:
            name (str): The name of the input node.

        Returns:
            dict: The input node configuration.

        """
        return {
            "type": "Input",
            "name": name,
            "outputs": [{"name": "document", "type": "Object"}],
        }

    def create_prompt_node(self, name):
        """
        Creates a parameterized prompt node configuration for a Bedrock flow.

        Args:
            name (str): The name of the prompt node.

        Returns:
            dict: The prompt node configuration.

        """
        return {
            "type": "Prompt",
            "name": name,
            "configuration": {
                "prompt": {
                    "sourceConfiguration": {
                        "promptReference": {
                            "promptId": self.prompt_id,
                            "promptVersion": self.prompt_version,
                        }
                    }
                }
            },
            "inputs": self.get_inputs_from_prompt(),
            "outputs": [{"name": "modelCompletion", "type": "String"}],
        }
        # return {
        #     "type": "Prompt",
        #     "name": name,
        #     "configuration": {
        #         "prompt": {
        #             "sourceConfiguration": {
        #                 "inline": {
        #                     "modelId": self.model_id,
        #                     "templateType": "TEXT",
        #                     "inferenceConfiguration": {
        #                         "text": self.inference_parameters
        #                     },
        #                     "templateConfiguration": {
        #                         "text": {
        #                             "text": self.prompt
        #                         }
        #                     }
        #                 }
        #             }
        #         }
        #     },
        #     "inputs": self.get_inputs_from_prompt(),
        #     "outputs": [
        #         {
        #             "name": "modelCompletion",
        #             "type": "String"
        #         }
        #     ]
        # }

    def get_inputs_from_prompt(self) -> list:
        """
        Extracts input variable names from the prompt that are wrapped in double curly braces.
        All inputs are of type String.

        Returns:
            list: A list of input dictionaries with name, type, and expression.
        """
        pattern = r"\{\{\s*(\w+)\s*\}\}"
        variables = re.findall(pattern, self.prompt)

        return [
            {"name": var, "type": "String", "expression": f"$.data.{var}"}
            for var in variables
        ]

    def create_output_node(self, name):
        """
        Creates an output node configuration for a Bedrock flow.

        The output node validates that the output from the last node is a string
        and returns it unmodified. The input name must be "document".

        Args:
            name (str): The name of the output node.

        Returns:
            dict: The output node configuration containing the output node:

        """
        return {
            "type": "Output",
            "name": name,
            "inputs": [{"name": "document", "type": "String", "expression": "$.data"}],
        }

    def create_flow(self, client, flow_name, flow_description, role_arn):
        """
        Creates the flow by connecting the nodes.
        Args:
            client: bedrock agent boto3 client.
            role_arn (str): Name for the new IAM role.
        Returns:
            dict: The response from the create_flow operation.
        """

        input_node = self.create_input_node("FlowInput")
        prompt_node = self.create_prompt_node("FlowPrompt")
        output_node = self.create_output_node("FlowOutput")

        # Create connections between the nodes
        connections = []

        #  First, create connections between the output of the flow
        # input node and each input of the prompt node.
        for prompt_node_input in prompt_node["inputs"]:
            connections.append(
                {
                    "name": "_".join(
                        [
                            input_node["name"],
                            prompt_node["name"],
                            prompt_node_input["name"],
                        ]
                    ),
                    "source": input_node["name"],
                    "target": prompt_node["name"],
                    "type": "Data",
                    "configuration": {
                        "data": {
                            "sourceOutput": input_node["outputs"][0]["name"],
                            "targetInput": prompt_node_input["name"],
                        }
                    },
                }
            )

        # Then, create a connection between the output of the prompt node and the input of the flow output node
        connections.append(
            {
                "name": "_".join([prompt_node["name"], output_node["name"]]),
                "source": prompt_node["name"],
                "target": output_node["name"],
                "type": "Data",
                "configuration": {
                    "data": {
                        "sourceOutput": prompt_node["outputs"][0]["name"],
                        "targetInput": output_node["inputs"][0]["name"],
                    }
                },
            }
        )

        flow_def = {
            "nodes": [input_node, prompt_node, output_node],
            "connections": connections,
        }

        # Create the flow.
        response = self._create_flow(
            client, flow_name, flow_description, role_arn, flow_def
        )

        return response

    def _create_flow(client, flow_name, flow_description, role_arn, flow_def):
        """
        Creates an Amazon Bedrock flow.

        Args:
        client: Amazon Bedrock agent boto3 client.
        flow_name (str): The name for the new flow.
        role_arn (str):  The ARN for the IAM role that use flow uses.
        flow_def (json): The JSON definition of the flow that you want to create.

        Returns:
            dict: The response from CreateFlow.
        """
        try:

            logger.info("Creating flow: %s.", flow_name)

            response = client.create_flow(
                name=flow_name,
                description=flow_description,
                executionRoleArn=role_arn,
                definition=flow_def,
            )

            logger.info(
                "Successfully created flow: %s. ID: %s", flow_name, {response["id"]}
            )

            return response

        except ClientError as e:
            logger.exception("Client error creating flow: %s", {str(e)})
            raise

        except Exception as e:
            logger.exception("Unexepcted error creating flow: %s", {str(e)})
            raise
