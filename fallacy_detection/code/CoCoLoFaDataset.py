from Dataset import Dataset
from datasets import Dataset as HuggingFaceDataset
import pandas as pd


class CoCoLoFaDataset(Dataset, name="cocolofa"):
    def __init__(self, name):
        self._input_key = "comment"
        self._label_key = "fallacy"
        self.input_file = f"datasets/{name}.json"
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
        return ["1" if label != "none" else "0" for label in raw_labels]
    
    def load(self):
        df = pd.read_json(self.input_file)

        all_comments = []
        for i, article in df.iterrows():
            all_comments.extend(article.get("comments", []))

        comments_df = pd.DataFrame(all_comments)
        return HuggingFaceDataset.from_pandas(comments_df)
