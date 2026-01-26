from Dataset import Dataset
from datasets import Dataset as HuggingFaceDataset
import pandas as pd


class ElecDeb60to20Dataset(Dataset, name="elecdeb60to20"):
    def __init__(self, name):
        self._input_key = "Context"
        self._label_key = "Label"
        self.input_file = f"datasets/{name}.csv"
        super().__init__(name)

    @property
    def input_key(self) -> str:
        return self._input_key
    
    @property
    def label_key(self) -> str:
        return self._label_key

    @property
    def labels(self):
        raw_labels = self.data[self.label_key]
        return ["1" if label else "0" for label in raw_labels]
    
    def load(self):
        df = pd.read_csv(self.input_file)
        return HuggingFaceDataset.from_pandas(df)
