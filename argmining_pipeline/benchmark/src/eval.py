import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, help="Step in argument mining process.", choices=["asd", "acc", "arc"], required=True)
    parser.add_argument("-i", "--input_file", type=str, help="Path to the input file.", required=True)
    return parser.parse_args()

def main():
    args = create_argparser()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.normpath(os.path.join(BASE_DIR, args.input_file))
    
    # Load data
    df = pd.read_csv(input_file)

    match args.task:
        case "asd":
            report = classification_report(df["gold_label"], df["pred"]) 
            print("ASD results:\n", report)
        case "acc":
            filtered_df = df.loc[df["component_type"].isin(["Claim", "Premise"])] 
            report = classification_report(filtered_df["component_type"].str.lower(), filtered_df["pred"]) 
            print("ACC results:\n", report)
        case "arc":
            # Map predictions to gold label categories
            mapping = {
                "inference": "support",
                "conflict": "attack",
                "none": "none"
            }
            df["pred_mapped"] = df["pred"].map(mapping)

            # Ignore 'rephrase' predictions
            df = df[df["pred"] != "rephrase"].copy()

            y_true = df["relation_type"]
            y_pred = df["pred_mapped"]
            labels = ["support", "attack", "none"]

            report = classification_report(
                y_true,
                y_pred,
                labels=labels,
                zero_division=0
            )
            print("ARC results:\n", report)

            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            cm_df = pd.DataFrame(cm, index=[f"True {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
            print(cm_df)

            # Plot confusion matrix as heatmap
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.tight_layout()
            plt.show()
        case _:
            print("Invalid task specification, exiting program.")
            return 1


if __name__ == "__main__":
    main()
