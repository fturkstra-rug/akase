from typing import Optional


def format_prompt(
    user_message: str,
    system_prompt: Optional[str] = None,
    examples: Optional[dict[str, str]] = None,
) -> str:
    prompt = "<|begin_of_text|>"

    # Add system instructions
    if system_prompt is not None:
        prompt += f"<|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|>"

    # Add in-context examples
    if examples is not None:
        for example, reply in examples.items():
            prompt += f"<|start_header_id|>user<|end_header_id|>{example}<|eot_id|>"
            prompt += f"<|start_header_id|>assistant<|end_header_id|>{reply}<|eot_id|>"

    # Add user prompt
    prompt += f"<|start_header_id|>user<|end_header_id|>{user_message}<|eot_id|>"

    # Add assistant token to start generation
    prompt += "<|start_header_id|>assistant<|end_header_id|>"

    return prompt


system_prompt = """You are the Proponent in a formal debate. Your role is to argue why the question is useful in analyzing the given argument.

The answer to a useful question can potentially invalidate or diminish the validity of the argument. The answer to a not useful question cannot, or is unlikely to, invalidate or diminish the validity of an argument.

The debate has three rounds:
Opening Statement: Present a clear case for why the question is critical. Does it expose a flaw in reasoning, challenge a hidden assumption, or question the logic of the inference?
Rebuttal: Respond directly to the Opponent’s objections. Defend the usefulness of the question and counter their critiques.
Closing Statement: Reinforce your argument. Emphasize why the question strengthens critical analysis of the original argument.
Be logical, concise, and persuasive in all your responses."""

user_message = """Debate Topic:
Argument: {{argument}}
Question: {{question}}

Debate So Far:
Introduction:
{{introduction}}

You are the Proponent.
This is your Opening Statement. Make your case for why the question is useful in critically evaluating the argument."""

formatted = format_prompt(user_message=user_message, system_prompt=system_prompt)
print(formatted)
