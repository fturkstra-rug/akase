import pandas as pd

class Dataset:
    _registry = {}

    def __new__(cls, name, *args, **kwargs):
        if cls is Dataset:
            if name not in cls._registry:
                raise ValueError(f"Unknown dataset: {name}")
            subclass = cls._registry[name]
            return super(Dataset, subclass).__new__(subclass)
        return super(Dataset, cls).__new__(cls)

    def __init_subclass__(cls, name=None):
        super().__init_subclass__()
        if name is not None:
            Dataset._registry[name] = cls

    def __init__(self, name: str):
        self.name = name
        self.data = self.load()

    def __len__(self) -> int:
        return len(self.data)
      
    @property
    def input_key(self) -> str:
        raise NotImplementedError

    @property
    def label_key(self) -> str:
        raise NotImplementedError

    @property
    def labels(self) -> list:
        raise NotImplementedError

    def load(self) -> pd.DataFrame:
        raise NotImplementedError
    