from Dataset import Dataset
from datasets import load_dataset


class RuFalDataset(Dataset, name="rufal"):
    def __init__(self, name):
        self._input_key = "text"
        self._label_key = "labels"
        self.input_file = "benmshultz/RuFal_fallacy_detection"
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
        return load_dataset(self.input_file, split="test")
