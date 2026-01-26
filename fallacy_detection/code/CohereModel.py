from Model import Model
import pandas as pd
from typing import Optional, Union

class CohereModel(Model, model_key="cohere"):
    def request_body(self, prompt: str, temperature: float=0, top_p: float=0.99, max_gen_len: int=3) -> dict:
        # return {
        #     "message": prompt,
        #     "temperature": temperature,
        #     "p": top_p,
        #     "k": 50,
        #     "max_tokens": max_gen_len
        # }
        return {
            "prompt": prompt,
            "temperature": temperature,
            "p": top_p,
            "k": 50,
            "max_tokens": max_gen_len
        }
    
    @classmethod
    def extract_output(cls, model_outputs_df: pd.DataFrame, size: str) -> pd.Series:
        # return model_outputs_df['modelOutput'].apply(lambda x: x["text"].strip()[0]) # for Command R
        outputs = model_outputs_df['modelOutput'].apply(lambda x: x['generations'][0]["text"].strip()[0])
        if size == 'small-2':
            outputs = outputs.str.translate(str.maketrans({'v': '0', 'f': '1'}))
        return outputs
    