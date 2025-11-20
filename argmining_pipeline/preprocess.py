#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed
from preprocessing import sentence_tokenize
from tqdm import tqdm
import logging
import pyarrow as pa
import re
import os
import time


# --- Setup logging ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


# --- Cleaning + tokenization ---
def clean_tokenize(text: str):
    """Clean HTML, normalize whitespace, and split into sentences."""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return []
    sents = sentence_tokenize(clean)
    return sents


# --- Process a batch of rows ---
# def process_batch(batch_df: pd.DataFrame, content_col: str):
#     cleaned = [clean_tokenize(txt) for txt in batch_df[content_col].astype(str)]
#     return pd.DataFrame({"preprocessed_sentences": [json.dumps(c) for c in cleaned]})


def process_batch(batch_df: pd.DataFrame, content_col: str):
    # Apply your cleaning/tokenization
    cleaned = [clean_tokenize(txt) for txt in batch_df[content_col].astype(str)]

    rows = []
    for doc_id, sentences in zip(batch_df["id"], cleaned):
        for sentence in sentences:
            text, start, end = sentence
            rows.append({
                "id": doc_id,
                "text": text,
                "start_index": start,
                "end_index": end,
            })

    return pd.DataFrame(rows)

# --- Arrow Dataset-based preprocessing ---
def preprocess_dataset(input_dir: Path, output_dir: Path, workers: int, batch_size: int):
    dataset = ds.dataset(
        [f for f in Path(input_dir).rglob("*.parquet")],
        format="parquet"
    )
    fragments = list(dataset.get_fragments())

    logger.info(f"Found {len(fragments)} Parquet fragments in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0

    # Process all fragments concurrently
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for frag in fragments:
            rel_path = Path(frag.path).relative_to(input_dir)
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                logger.info(f"Skipping {rel_path} (already processed).")
                continue
            futures[pool.submit(process_fragment, frag, out_path, batch_size)] = frag

        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing fragments"):
            frag = futures[f]
            try:
                f.result()
                processed += 1
            except Exception as e:
                logger.error(f"Error processing {frag.path}: {e}")

    logger.info(f"Finished preprocessing {processed} fragments.")


# --- Per-fragment function ---
def process_fragment(fragment, out_path: Path, batch_size: int):
    table = fragment.to_table()
    df = table.to_pandas()

    # Pick content column
    if "main_content" in df.columns:
        content_col = "main_content"
    elif "plain_text" in df.columns:
        content_col = "plain_text"
    else:
        logger.warning(f"No content column in {fragment.path}. Skipping.")
        return

    results = []
    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start:start + batch_size]
        results.append(process_batch(batch_df, content_col))

    df_out = pd.concat(results, ignore_index=True)
    pq.write_table(pa.Table.from_pandas(df_out), out_path)
    logger.info(f"Saved {len(df_out)} rows to {out_path}")


# --- Main driver ---
def main(input_dir: str, output_dir: str, workers: int, batch_size: int):
    start = time.time()

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    preprocess_dataset(input_dir, output_dir, workers, batch_size)

    elapsed = time.time() - start
    logger.info("=" * 80)
    logger.info("PREPROCESSING COMPLETE")
    logger.info(f"Total time: {elapsed / 60:.2f} minutes")
    logger.info(f"Workers: {workers}")
    logger.info(f"Batch size: {batch_size}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-performance Parquet preprocessing (Arrow Dataset + multiprocessing)")
    parser.add_argument("--input_dir", required=True, help="Input directory with parquet files")
    parser.add_argument("--output_dir", required=True, help="Output directory for cleaned parquet files")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel processes (default: all cores)")
    parser.add_argument("--batch_size", type=int, default=5000, help="Batch size for in-memory processing")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.workers, args.batch_size)
