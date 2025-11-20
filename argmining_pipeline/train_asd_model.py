from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import argparse
from pathlib import Path
from datasets import Dataset
import numpy as np

def load_dataset(input_dir: str, downsample:bool):

    # Read data
    df = pd.concat(pd.read_csv(f, sep='\t', quoting=3) for f in input_dir.glob('*.tsv'))

    # Map labels: NoArgument -> 0, everything else -> 1
    df['label'] = (df['annotation'] != 'NoArgument').astype(int)

    # Downsample majority class to match minority, recombine and shuffle
    if downsample:
        df_minority = df[df['label'] == 1]
        df_majority = df[df['label'] == 0].sample(n=len(df_minority), random_state=42)
        df = pd.concat([df_minority, df_majority]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Create splits and convert to HF Datasets 
    train_df = df.loc[df['set'] == 'train']
    val_df = df.loc[df['set'] == 'val']
    test_df = df.loc[df['set'] == 'test']

    dataset = {
        "train": Dataset.from_pandas(train_df),
        "val": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    }

    return dataset


def main(input_dir: str, output_dir: str, epochs: int = 3, batch_size: int = 8, downsample: bool = False):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    print("Loading dataset...")
    dataset = load_dataset(input_dir, downsample)
    print("Loaded dataset successfully!")

    model_name = "distilbert/distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    # Tokenize
    train_dataset = dataset['train'].map(tokenize_function, batched=True)
    val_dataset = dataset['val'].map(tokenize_function, batched=True)
    test_dataset = dataset['test'].map(tokenize_function, batched=True)

    # Keep only necessary columns
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in ["input_ids", "attention_mask", "label"]])
    val_dataset = val_dataset.remove_columns([col for col in val_dataset.column_names if col not in ["input_ids", "attention_mask", "label"]])
    test_dataset = test_dataset.remove_columns([col for col in test_dataset.column_names if col not in ["input_ids", "attention_mask", "label"]])

    # def compute_metrics(pred):
    #     preds = np.argmax(pred.predictions, axis=-1)
    #     return {"accuracy": (preds == pred.label_ids).mean()}

    def compute_metrics(pred):
        preds = np.argmax(pred.predictions, axis=-1)

        accuracy = accuracy_score(pred.label_ids, preds)

        precision, recall, f1, _ = precision_recall_fscore_support(
            pred.label_ids,
            preds,
            average="weighted"
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # Train model
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=output_dir / "logs",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate on test set
    results = trainer.evaluate(test_dataset)
    print("Test set performance:", results)

    # 1️Save final model
    trainer.save_model(output_dir / "final_model")
    tokenizer.save_pretrained(output_dir / "final_model")
    print(f"Model saved at {output_dir / 'final_model'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Path to input data files.")
    parser.add_argument("--output_dir", "-o", type=str, default="./classification_output", help="Where to save model/checkpoints.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device.")
    parser.add_argument("--downsample", action='store_true', help="Whether to downsample the majority class.")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.epochs, args.batch_size, args.downsample)
