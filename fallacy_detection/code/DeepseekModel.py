from Model import Model
from typing import Optional, Union
import pandas as pd

class DeepseekModel(Model, model_key="deepseek"):
    def request_body(self, prompt: str, temperature: float=0, top_p: float=1, top_k: int=50, max_tokens: int=512) -> dict:
        # parameters = {
        #     "temperature": temperature, 
        #     "top_p": top_p,
        #     "max_new_tokens": max_tokens
        # }
        # return {
        #     "inputs": prompt,
        #     "parameters": parameters
        # }
        return {
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
    
    
    @classmethod
    def extract_output(cls, model_outputs_df: pd.DataFrame, size: str) -> pd.Series:
        return model_outputs_df['modelOutput'].apply(lambda x: x['choices']["text"].strip()[0])

    def format_prompt(
        self,
        message: Union[str, list[dict]],
        system_prompt: Optional[str] = None,
    ) -> str:
        prompt = "<|begin_of_sentence|>"

        # Add system instructions
        if system_prompt is not None:
            prompt += f"<|System|>{system_prompt}"

        # Add in-context examples
        if isinstance(message, list):
            for msg in message:
                text = msg["content"][0]["text"]
                role = msg["role"]

                if role == "user":
                    prompt += f"<|User|>{text}"
                elif role == "assistant":
                    prompt += f"<|Assistant|>{text}<|end_of_sentence|>"
        else:
            prompt += f"<|User|>{message}"
            
        prompt += "<|Assistant|>\n"

        return prompt
