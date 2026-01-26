from Dataset import Dataset
from datasets import Dataset as HuggingFaceDataset
import pickle

class MMArgFallacyDataset(Dataset, name="mmargfallacy"):
    def __init__(self, name):
        self._input_key = "snippet"
        self._label_key = "fallacy"
        self.input_file = f"datasets/{name}.pkl"
        super().__init__(name)

    @property
    def input_key(self) -> str:
        return self._input_key
    
    @property
    def label_key(self) -> str:
        return self._label_key

    @property
    def labels(self) -> list:
        raw_labels = self.data[self.label_key]
        return ["1" if label else "0" for label in raw_labels]
    
    def load(self) -> HuggingFaceDataset:
        with open(self.input_file, "rb") as f:
            data = pickle.load(f)

        # Apply to the column
        data.drop(columns=[
            "dialogue_whisper_indexes",
            "snippet_tokens",
            "snippet_whisper_indexes"],
            inplace=True
            )
        
        return HuggingFaceDataset.from_pandas(data)
    
    
