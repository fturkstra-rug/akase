from Dataset import Dataset
from datasets import Dataset as HuggingFaceDataset
import pandas as pd
import re


class TestDataset(Dataset, name="test"):
    def __init__(self, name):
        self._input_key = "text"
        self._label_key = None
        self.input_file = "deliberation/general_round_1_credibility/model_outputs.jsonl"
        super().__init__(name)

    @property
    def input_key(self) -> str:
        return self._input_key
    
    @property
    def label_key(self) -> str:
        return self._label_key

    @property
    def labels(self) -> list:
        return self.data[self.label_key]
    
    def load(self) -> HuggingFaceDataset:
        df = pd.read_json(self.input_file, lines=True)

        # Extract deliberation so far
        df['old'] = df['modelInput'].apply(
            lambda x: re.search(r'<deliberation>(.*?)</deliberation>', x['prompt'], re.DOTALL).group(1)
        )

        # Add previous model output to deliberation
        df["new"] = df['modelOutput'].apply(lambda x: f"<deliberator_3>{x['generation'].strip()}</deliberator_3>")

        df['text'] = df['old'] + df['new']

        return HuggingFaceDataset.from_pandas(df)
    