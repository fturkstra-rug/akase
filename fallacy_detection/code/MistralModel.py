from Model import Model
from typing import Optional, Union
import pandas as pd

class MistralModel(Model, model_key="mistral"):
    def request_body(self, prompt: str, temperature: float=0, top_p: float=1, top_k: int=50, max_tokens: int=2) -> dict:
        return {
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens
        }
    
    @classmethod
    def extract_output(cls, model_outputs_df: pd.DataFrame, size: str) -> pd.Series:
        if size == "large":
            return model_outputs_df['modelOutput'].apply(lambda x: x['choices'][0]['message']['content'].strip().lstrip('"')[0])
        
        for row in model_outputs_df['modelOutput']:
            if not row or 'outputs' not in row or not row['outputs']:
                print(row)

        return model_outputs_df['modelOutput'].apply(lambda x: x['outputs'][0]['text'].strip()[0])

    def format_prompt(
        self,
        message: Union[str, list[dict]],
        system_prompt: Optional[str] = None,
    ) -> str:
        prompt = "<s>"

        # Add system instructions
        if system_prompt is not None:
            prompt += f"[INST] {system_prompt} [/INST]"

        # Add in-context examples
        if isinstance(message, list):
            for msg in message:
                text = msg["content"][0]["text"]
                role = msg["role"]

                if role == "user":
                    prompt += f"[INST] {text} [/INST]"
                elif role == "assistant":
                    prompt += f"{text}</s>"
        else:
            prompt += f"[INST] {message} [/INST]"

        return prompt
