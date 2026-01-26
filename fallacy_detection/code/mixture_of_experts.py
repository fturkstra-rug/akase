from pathlib import Path
import pandas as pd
import itertools
from Model import Model
from MistralModel import MistralModel
from MetaModel import MetaModel
from CohereModel import CohereModel
from Dataset import Dataset
from MafaldaDataset import MafaldaDataset
from LogicDataset import LogicDataset
from LogicClimateDataset import LogicClimateDataset
from RuFalDataset import RuFalDataset
from ElecDeb60to20 import ElecDeb60to20Dataset
from CoCoLoFaDataset import CoCoLoFaDataset
from NonFallaciousDataset import NonFallaciousDataset
import random
from tqdm import tqdm

experiments = Path("experiments")

SPLIT = "test"

class MiniModel:
    def __init__(self, model, model_class):
        self.technique, self.name, self.size = model.split('_')
        self.model_class = model_class
        df = pd.read_json(f'experiments/{model}/model_outputs_{SPLIT}.jsonl', lines=True)
        df['recordId'] = df['recordId'].astype(int)
        self.df = df.sort_values(by='recordId')
        self.predictions = self.model_class.extract_output(self.df, self.size)

models = {
    # "zeroshot_llama_small": MiniModel("zeroshot_llama_small", MetaModel),
    "zeroshot_llama_medium": MiniModel("zeroshot_llama_medium", MetaModel),
    # "zeroshot_mistral_small": MiniModel("zeroshot_mistral_small", MistralModel),
    # "zeroshot_mistral_medium": MiniModel("zeroshot_mistral_medium", MistralModel),
    # "zeroshot_cohere_small": MiniModel("zeroshot_cohere_small", CohereModel),
    # "zeroshot_cohere_medium": MiniModel("zeroshot_cohere_medium", CohereModel),
    # "fewshot_llama_small": MiniModel("fewshot_llama_small", MetaModel),
    "fewshot_llama_medium": MiniModel("fewshot_llama_medium", MetaModel),
    # "fewshot_mistral_small": MiniModel("fewshot_mistral_small", MistralModel),
    # "fewshot_mistral_medium": MiniModel("fewshot_mistral_medium", MistralModel),
    # "fewshot_cohere_small": MiniModel("fewshot_cohere_small", CohereModel),
    # "fewshot_cohere_medium": MiniModel("fewshot_cohere_medium", CohereModel),
    # "cot_llama_small": MiniModel("cot_llama_small", MetaModel),
    # "cot_llama_medium": MiniModel("cot_llama_medium", MetaModel),
    # "cot_mistral_small": MiniModel("cot_mistral_small", MistralModel),
    # "cot_mistral_medium": MiniModel("cot_mistral_medium", MistralModel),
    # "cot_cohere_small-2": MiniModel("cot_cohere_small-2", CohereModel),
    # "cot_cohere_medium-2": MiniModel("cot_cohere_medium-2", CohereModel),
}

# Extract labels from the datasets
# dataset_names = ["mafalda", "logic", "logicclimate", "rufal", "elecdeb60to20", "cocolofa", "non-fallacious"]
# datasets = [Dataset(name) for name in dataset_names]
    
# labels = []
# for dataset in datasets:
#     labels.extend(dataset.labels)

dataset = pd.read_json(f"datasets/{SPLIT}.jsonl", lines=True)
dataset = dataset.sort_values(by='recordId')
labels = dataset['label'].astype(str).tolist()
deliberation_votes = pd.read_json(f"deliberation/{SPLIT}_round_1_conclusion/deliberation_vote.jsonl", lines=True)

# 3060 combinations if we also mix up the techniques
# 20 combinations if we consider per technique

combinations = list(itertools.combinations(models.keys(), r=2))
results = []
random.seed(42)

disagree_ids = []

for combination in tqdm(combinations, desc="Combinations"):
    predictions = [models[name].predictions for name in combination]

    correct = 0
    disagree_count = 0
    correct_agreement = 0

    i = 0
    final_predictions = []

    for a, b, label in zip(*predictions, labels):

        if a not in ('0', '1') or b not in ('0', '1'):
            # if a not in ('0', '1') and b in ('0', '1'):
            #     final_predictions.append(b)
            #     if b == label:
            #         correct += 1
            # elif a in ('0', '1') and b not in ('0', '1'):
            #     final_predictions.append(a)
            #     if a == label:
            #         correct += 1
            # else:
            disagree_count += 1
            disagree_ids.append(dataset['recordId'][i])
            final_predictions.append('x')
        else:
            if a == b:
                final_predictions.append(a)
                if a == label:
                    correct += 1
                    correct_agreement += 1
            else:
                disagree_count += 1
                disagree_ids.append(dataset['recordId'][i])
                # c = 'x'

                if False: #dataset['recordId'][i] in deliberation_votes['recordId'].values:
                    # Get the vote from the deliberation votes
                    c = deliberation_votes[deliberation_votes['recordId'] == dataset['recordId'][i]]['prediction'].values[0]
                    c = str(c)
                else:
                    print('something went wrong')
                    c = random.choice([a, b])

                final_predictions.append(c)
                if c == label:
                    correct += 1

        i += 1

    # print(len(dataset['recordId']), len(labels), len(final_predictions))

    # df = pd.DataFrame({"recordId": dataset['recordId'], "label": labels, "prediction": final_predictions})
    # df.to_json(f"deliberation/test_round_1_conclusion/majority_vote.jsonl", orient="records", lines=True)
    # print(df['prediction'].value_counts())
    # exit()

    accuracy = correct / len(labels)
    # print(len(disagree_ids))
    disagree_df =dataset[dataset['recordId'].isin(disagree_ids)]
    # print(len(disagree_df))
    disagree_df.to_json(f"datasets/disagreements_{SPLIT}.jsonl", orient="records", lines=True)

    # print(technique, model, size, accuracy)
    results.append({"combination": combination, "accuracy": f"{accuracy * 100:.1f}", "disagreements": disagree_count, "correct_agree": correct_agreement})


df = pd.DataFrame(results)
df = df.sort_values('accuracy', ascending=False, ignore_index=True)
# print(len(df))
print(df.head(20))



    
