from Dataset import Dataset
from datasets import Dataset as HuggingFaceDataset
import json

class MafaldaDataset(Dataset, name="mafalda"):
    def __init__(self, name):
        self._input_key = "text"
        self._label_key = "labels"
        self.input_file = f"datasets/{name}.jsonl"
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
        with open(self.input_file, "r") as f:
            data = [json.loads(line) for line in f]
            # Labels switch between int and str apparently, so we need to convert them to str
            # data["labels"] = data["labels"].apply(lambda lst: [[str(x) for x in sublist] for sublist in lst])

            for entry in data:
                entry["labels"] = [[str(x) for x in sublist] for sublist in entry["labels"]]

        return HuggingFaceDataset.from_list(data)
    
    
