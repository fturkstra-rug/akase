import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import argparse
from datasets import Dataset
import gc


@torch.inference_mode()
def predict_dataset(model, dataset, batch_size=256, device="cuda", use_fp16=False):
    """Run batched inference on a tokenized Hugging Face Dataset."""
    preds = []
    
    for batch in dataset.iter(batch_size=batch_size):
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }

        with torch.autocast("cuda", dtype=torch.float16) if use_fp16 else torch.no_grad():
            logits = model(**inputs).logits
            batch_preds = torch.argmax(logits, dim=-1)
        preds.extend(batch_preds.cpu().tolist())

    return preds


def process_parquet_file(file_path, model, tokenizer, output_file, batch_size=256, device="cuda", use_fp16=False):
    """Run inference on a single parquet file and save output."""
    try:
        df = pd.read_parquet(file_path)

        if "text" not in df.columns:
            print(f"Skipping {file_path}: missing 'text' column.")
            return

        # Convert to Hugging Face Dataset for paralllel tokenization
        dataset = Dataset.from_pandas(df)

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=128, 
            )

        dataset = dataset.map(tokenize_function, batched=True, num_proc=4, desc=None)

        # Keep only model-relevant columns
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        # Run inference
        preds = predict_dataset(model, dataset, batch_size, device, use_fp16)

        # Add predictions back to DataFrame
        df["is_arg"] = preds

        # Save results (mirror output structure)
        os.makedirs(Path(output_file).parent, exist_ok=True)
        df.to_parquet(output_file, index=False)

        # clean up
        del dataset
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main(input_dir, output_dir, model_dir, batch_size=256, use_fp16=True):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    model = torch.compile(model)  # JIT optimize for A100
    print("Successfully loaded model.")

    # Collect all parquet files recursively
    parquet_files = list(input_dir.rglob("*.parquet"))
    print(f"Found {len(parquet_files):,} parquet files.")

    for file_path in tqdm(parquet_files, desc="Inferencing", unit="file"):

        relative_path = file_path.relative_to(input_dir)
        output_file = output_dir / relative_path

        # Skip if already processed
        if output_file.exists():
            continue

        process_parquet_file(
            file_path=file_path,
            model=model,
            tokenizer=tokenizer,
            output_file=output_file,
            batch_size=batch_size,
            device=device,
            use_fp16=use_fp16
        )

    print("Inference complete. All files processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Input directory containing parquet files.")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Where to save output parquet files.")
    parser.add_argument("--model_dir", "-m", type=str, required=True, help="Path to fine-tuned model directory.")
    parser.add_argument("--batch_size", "-b", type=int, default=512, help="Batch size for inference.")
    parser.add_argument("--no_fp16", action="store_true", help="Disable mixed precision (FP16).")
    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        use_fp16=not args.no_fp16,
    )
