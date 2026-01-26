import pandas as pd
from MetaModel import MetaModel
import random
from Dataset import Dataset
from MafaldaDataset import MafaldaDataset
from LogicDataset import LogicDataset
from LogicClimateDataset import LogicClimateDataset
from RuFalDataset import RuFalDataset
from ElecDeb60to20 import ElecDeb60to20Dataset
from CoCoLoFaDataset import CoCoLoFaDataset
from NonFallaciousDataset import NonFallaciousDataset


df1 = pd.read_json("deliberation/general_round_1_emotion_vote/model_outputs.jsonl", lines=True)
df2 = pd.read_json("deliberation/general_round_1_logic_vote/model_outputs.jsonl", lines=True)
df3 = pd.read_json("deliberation/general_round_1_credibility_vote/model_outputs.jsonl", lines=True)


predictions1 = MetaModel.extract_output(df1, "medium")
predictions2 = MetaModel.extract_output(df2, "medium")
predictions3 = MetaModel.extract_output(df3, "medium")

dataset_names = ["mafalda", "logic", "logicclimate", "rufal", "elecdeb60to20", "cocolofa", "non-fallacious"]
datasets = [Dataset(name) for name in dataset_names]

labels = []
for dataset in datasets:
    labels.extend(dataset.labels)

votes = []
invalid = 0
for predictions in zip(predictions1, predictions2, predictions3):
    valid_predictions = [p for p in predictions if p in ('0', '1')]
    invalid += len(predictions) - len(valid_predictions)

    if not valid_predictions:
        vote = 'x'
    else:
        count_0 = valid_predictions.count('0')
        count_1 = valid_predictions.count('1')

        if count_0 > count_1:
            vote = '0'
        elif count_1 > count_0:
            vote = '1'
        else:
            vote = random.choice(valid_predictions)

    votes.append(vote)

correct = 0
for label, pred in zip(labels, votes):
    if pred in ('0', '1') and pred == label:
        correct += 1

accuracy = correct / len(labels)

# print(technique, model, size, accuracy)
print("Accuracy", f"{accuracy * 100:.1f}")
print("Invalid", invalid)
