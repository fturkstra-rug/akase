import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pred_file", type=str, help="Path to the predictions file.", required=True)
parser.add_argument("-g", "--gold_file", type=str, help="Path to the golden labels file.", required=True)
args = parser.parse_args()


pred_file = args.pred_file
gold_file = args.gold_file

with open(pred_file, "r") as f:
    pred_data = json.load(f)

with open(gold_file, "r") as f:
    gold_data = json.load(f)

assert set(pred_data.keys()) == set(gold_data.keys()), "Sets contain different ids."
interventions = gold_data.keys()

total_correct = 0
total = 0

for intervention in interventions:
    predictions = pred_data[intervention]["cqs"]
    gold_labels = gold_data[intervention]["cqs"]

    for pred, gold in zip(predictions, gold_labels):
        gold_label = gold["label"]
        prediction = pred["label"]
        
        total_correct += (gold_label == "Useful") == (prediction == "Useful")
        total += 1

print(f"Total correct: {total_correct}")
print(f"Total: {total}")
print(f"Accuracy: {round(total_correct / total, 4)}")
