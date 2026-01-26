from Dataset import Dataset
from MafaldaDataset import MafaldaDataset
from LogicDataset import LogicDataset
from LogicClimateDataset import LogicClimateDataset
from RuFalDataset import RuFalDataset
from ElecDeb60to20 import ElecDeb60to20Dataset
from CoCoLoFaDataset import CoCoLoFaDataset
from NonFallaciousDataset import NonFallaciousDataset
import pandas as pd
from sklearn.model_selection import train_test_split


dataset_names = ["mafalda", "logic", "logicclimate", "rufal", "elecdeb60to20", "cocolofa", "non-fallacious"]
datasets = [Dataset(name) for name in dataset_names]

labels = []
names = []
arguments = []
record_ids = []

current_id = 1

for dataset in datasets:
    names.extend([dataset.name] * len(dataset.labels))
    labels.extend(dataset.labels)
    arguments.extend(dataset.data[dataset.input_key])
    record_ids.extend(range(current_id, current_id + len(dataset.labels)))

    current_id += len(dataset.labels)

df = pd.DataFrame(zip(arguments, labels, names, record_ids), columns=["argument", "label", "dataset", "recordId"])

# 20% validation, 80% test, stratified on label
df_test, df_val = train_test_split(
    df,
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

df_test.to_json("datasets/test.jsonl", orient="records", lines=True)
df_val.to_json("datasets/val.jsonl", orient="records", lines=True)
