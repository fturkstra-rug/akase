from mamkit.data.datasets import MMUSEDFallacy, InputMode
from pathlib import Path
import numpy as np

def load_data():
    base_data_path = Path(__file__).parent.parent.resolve().joinpath('data')

    # MM-USED-fallacy dataset
    loader = MMUSEDFallacy(
        task_name='afc', # Choose between 'afc' or 'afd'               
        input_mode=InputMode.TEXT_ONLY,
        base_data_path=base_data_path
    )
    split_info = loader.get_splits('mm-argfallacy-2025')

    return loader, split_info


if __name__ == "__main__":
    dataset, splits = load_data()

    # dataset.data is a dataframe
    print(splits[0])

    for split_info in splits:
        labels = np.array([
            label if label is not None else "None"
            for label in split_info.train.labels
        ])
        unique, counts = np.unique(labels, return_counts=True)

        for label, count in zip(unique, counts):
            print(f"Label: {label}, Count: {count}")