from Model import Model
from typing import Optional

class AnthropicModel(Model, model_key="anthropic"):
    def request_body(self, prompt: str, temperature: float=1, top_p: float=1, top_k: int=250, max_tokens_to_sample: int=200, stop_sequences: Optional[list] = None) -> dict:
        return {
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens_to_sample": max_tokens_to_sample,
            "stop_sequences": stop_sequences or ["\n\nHuman:"]
        }
