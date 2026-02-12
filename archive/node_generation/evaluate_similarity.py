import argparse
from create_embeddings import embed
import torch.nn.functional as F
import pandas as pd


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', help="File with test set predictions", required=True, type=str)
    parser.add_argument('--predictions_inverted', help="File with the inverted test set predictions", required=True, type=str)
    parser.add_argument('--labels', help="File with the test set gold labels", required=True, type=str)
    parser.add_argument('--model', help="Model that is being evaluated", required=True, type=str, choices=["t5-base", "bart-base"])
    return parser.parse_args()


def main():
    args = create_arg_parser()

    with open(args.predictions, 'r') as f1:
        predictions = f1.readlines()

    with open(args.predictions_inverted, 'r') as f2:
        inverted = f2.readlines()

    with open(args.labels, 'r') as f3:
        labels = f3.readlines()

    print(len(predictions))
    print(len(inverted))
    print(len(labels))

    assert len(predictions) == len(inverted) == len(labels), "Unequal lengths... :("

    batch_size = 4
    predictions_embeddings = embed(predictions, batch_size)
    inverted_embeddings = embed(inverted, batch_size)
    labels_embeddings = embed(labels, batch_size)

    predictions_embeddings_norm = F.normalize(predictions_embeddings, p=2, dim=1)
    inverted_embeddings_norm = F.normalize(inverted_embeddings, p=2, dim=1)
    labels_embeddings_norm = F.normalize(labels_embeddings, p=2, dim=1)

    predictions_inverted_sim = (predictions_embeddings_norm * inverted_embeddings_norm).sum(dim=1)  # Shape: [N]
    predictions_labels_sim = (predictions_embeddings_norm * labels_embeddings_norm).sum(dim=1)  # Shape: [N]
    inverted_label_sim = (inverted_embeddings_norm * labels_embeddings_norm).sum(dim=1)  # Shape: [N]

    data = {
        "predictions": predictions,
        "inverted": inverted,
        "labels": labels,
        "predictions_inverted_sim": predictions_inverted_sim,
        "predictions_labels_sim": predictions_labels_sim,
        "inverted_labels_sim": inverted_label_sim,
    }
    df = pd.DataFrame(data)
    print(df)
    df.to_csv(f'evaluate_similarity_{args.model}.csv')


if __name__ == "__main__":
    main()
