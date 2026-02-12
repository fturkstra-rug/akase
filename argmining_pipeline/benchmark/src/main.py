import pandas as pd
import argparse
from argmining.sentence_detection import SentenceDetector
from argmining.component_classification import ComponentClassifier
from argmining.relation_classification import CustomRelationClassifier
import os


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, help="Step in argument mining process.", choices=["asd", "acc", "arc"], required=True)
    parser.add_argument("-i", "--input_file", type=str, help="Path to the input file.", required=True)
    parser.add_argument("-o", "--output_file", type=str, help="Path to the output file.", required=True)
    return parser.parse_args()

def main():
    args = create_argparser()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.normpath(os.path.join(BASE_DIR, args.input_file))

    # Load data
    df = pd.read_csv(input_file) # type: ignore
    df = df.dropna()
    print(f"Loaded file '{input_file}' with {len(df)} rows.")

    # Load task-specific model and data.
    match args.task:
        case 'asd':
            model = SentenceDetector()
            model.load_or_train(force_train=False)
            input_data = df["text"].to_list()
        case 'acc':
            model = ComponentClassifier()
            model.load_or_train(force_train=False)
            input_data = df["text"].to_list()
        case 'arc':
            model = CustomRelationClassifier()
            input_data = [(row.source_text, row.target_text) for _, row in df.iterrows()]
        case _:
            print("Invalid task specification, exiting program.")
            return 1

    # Get model predictions
    preds: list[bool]
    preds, _ = model.predict(input_data) # type: ignore
    
    # Save predictions
    df["pred"] = preds

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.normpath(os.path.join(BASE_DIR, args.output_file))

    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
