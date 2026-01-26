import pandas as pd
import json
import numpy as np

def create_analysis_file():
    # Load the test set with 34 interventions
    with open("data/test.json", 'r') as f:
        data = json.load(f)

    # Load the 10 generated questions per intervention
    generated_questions = pd.read_json("data/test_gen_cqs.jsonl", lines=True)
    generated_questions = generated_questions['modelOutput'].apply(
        lambda x: [line.strip() for line in x['generation'].strip().split('\n') if line.strip()]
    )

    # Load the single prompt results
    with open("data/test_run1.json", 'r') as f1:
        prm_results = json.load(f1)
        run1 = pd.DataFrame(prm_results.values())["cqs"]

    # Load the deliberation results
    with open("data/test_run2.json", 'r') as f2:
        dlb_results = json.load(f2)
        run2 = pd.DataFrame(dlb_results.values())["cqs"]

    # Load the debate results
    with open("data/test_run3.json", 'r') as f3:
        dbt_results = json.load(f3)

        # Debate results are in a different order from the test set so re-sort
        ordered_keys = list(data.keys())
        dbt_results = {k: dbt_results[k] for k in ordered_keys if k in dbt_results}
        run3 = pd.DataFrame(dbt_results.values())["cqs"]

    # Add the generated questions and classification results to each intervention
    for key, entry in data.items():
        idx = list(data.keys()).index(key)
        cqs = generated_questions[idx]
        gold_labels = ["unknown" for _ in cqs]

        run1_answer = run1[idx]
        run2_answer = run2[idx]
        run3_answer = run3[idx]

        for answer in [run1[idx], run2[idx], run3[idx]]:
            for id_cq_label in answer:
                idx = cqs.index(id_cq_label["cq"])
                id_cq_label["idx"] = idx

                label = id_cq_label["label"].lower()
                if gold_labels[idx] == label:
                    continue
                elif gold_labels[idx] == "unknown" or gold_labels[idx] == "not_able_to_evaluate" or label == "not_able_to_evaluate":
                    gold_labels[idx] = label
                else:
                    print("Warning: multiple gold-labels.") # does not happen
                    gold_labels[idx] += "|" + label

        entry["cqs"] = cqs
        entry["gold_labels"] = gold_labels
        entry["run1"] = run1_answer
        entry["run2"] = run2_answer
        entry["run3"] = run3_answer

    with open("analysis.json", "w") as output_file:
        json.dump(data, output_file, indent=4)

    return data

def get_agreement(data, aggregated=True):
    prm_dlb_agreements = []
    prm_dbt_agreements = []
    dlb_dbt_agreements = []
    prm_dlb_dbt_agreements = []

    # Check agreement
    for entry in data.values():
        prm_labels = set([item["idx"] for item in entry["run1"]])
        dlb_labels = set([item["idx"] for item in entry["run2"]])
        dbt_labels = set([item["idx"] for item in entry["run3"]])

        prm_dlb_agreement = prm_labels.intersection(dlb_labels)
        prm_dbt_agreement = prm_labels.intersection(dbt_labels)
        dlb_dbt_agreement = dlb_labels.intersection(dbt_labels)
        prm_dlb_dbt_agreement = prm_labels & dlb_labels & dbt_labels

        prm_dlb_agreements.append(len(prm_dlb_agreement))
        prm_dbt_agreements.append(len(prm_dbt_agreement))
        dlb_dbt_agreements.append(len(dlb_dbt_agreement))
        prm_dlb_dbt_agreements.append(len(prm_dlb_dbt_agreement))

        # Print agreement per intervention
        if not aggregated:
            print(f"{entry['intervention_id']}:")
            print(f"  PRM ∩ DLB: {len(prm_dlb_agreement)}")
            print(f"  PRM ∩ DBT: {len(prm_dbt_agreement)}")
            print(f"  DLB ∩ DBT: {len(dlb_dbt_agreement)}")
            print(f"  PRM ∩ DLB ∩ DBT: {len(prm_dlb_dbt_agreement)}\n")
    
    # Print aggregated agreement scores on all interventions
    if aggregated:
        print(f"Average agreement:")
        print(f"  PRM ∩ DLB: {np.mean(prm_dlb_agreements)}")
        print(f"  PRM ∩ DBT: {np.mean(prm_dbt_agreements)}")
        print(f"  DLB ∩ DBT: {np.mean(dlb_dbt_agreements)}")
        print(f"  PRM ∩ DLB ∩ DBT: {np.mean(prm_dlb_dbt_agreements)}\n")


def check_gold_labels(data, aggregate=True):
    label_counts = {label: 0 for label in ["unknown", "useful", "unhelpful", "invalid", "not_able_to_evaluate"]}
    for entry in data.values():
        for l in entry["gold_labels"]:
            label_counts[l.lower()] += 1

    print(sum(label_counts.values())) # should be 340
    print(label_counts)

def analyze_prm_output():
    df = pd.read_json("data/test_prm_output.jsonl", lines=True)
    df = df.iloc[:34]

    model_outputs = df['modelOutput'].apply(lambda x: x['generation'].strip().split('\n'))
    
    useful_counts = []
    for output in model_outputs:
        useful_count = 0
        for label in eval(output.pop()):
            useful_count += label == "useful"

        useful_counts.append(useful_count)

    print(np.mean(useful_counts))


if __name__ == "__main__":
    data = create_analysis_file()
    
    # get_agreement(data)

    check_gold_labels(data)

    total_labels = 0
    useful_labels = 0

    for intervention in data.values():
        labels = [l for l in intervention['gold_labels'] if l.lower() != "unknown"]
        useful = [l for l in labels if l.lower() == "useful"]
        unhelpful = [l for l in labels if l.lower() == "unhelpful"]
        print(f"{intervention['intervention_id']}:\t\t{len(useful)} / {len(labels)} & {len(unhelpful)}")
        total_labels += len(labels)
        useful_labels += len(useful)

    print(f"{useful_labels} / {total_labels} = {useful_labels / total_labels}")
