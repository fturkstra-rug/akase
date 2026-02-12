import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import argparse
from huggingface_hub import login
import os
import time
import logging
import sys
from datetime import timedelta

# How to load the memory map
# x = np.memmap("query_embeddings.memmap", dtype=np.float32, mode="r", shape=(31156, 4096))
# where 31156 is the total number of embeddings and 4096 is the number of embedding dimensions.

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("embedding_process.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    start_time = time.time()
    
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, help="Path to the input file.", required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("-o", "--output_file", type=str, default="query_embeddings.npy", help="Output file name")
    parser.add_argument("-c", "--checkpoint_interval", type=int, default=1000, help="Save checkpoint after processing this many queries")
    parser.add_argument("-t", "--token", type=str, help="Hugging Face token")
    args = parser.parse_args()
    
    # Login to Hugging Face if token provided
    if args.token:
        login(token=args.token)
    
    # Check if GPU is available and set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if device == "cuda":
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    
    # Load model
    logger.info("Loading model...")
    model = SentenceTransformer(
        "Alibaba-NLP/gte-Qwen1.5-7B-instruct", 
        trust_remote_code=True,
        device=device
    )
    model.max_seq_length = 8192
    
    # Read queries from json
    logger.info(f"Reading queries from {args.input_file}")
    try:
        df = pd.read_json(args.input_file)
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return
    
    # Ensure "motion" and "uuid" columns exist
    if "motion" not in df.columns:
        logger.error(f"Column 'motion' not found in {args.input_file}. Please check the file structure.")
        return
    
    queries = df["motion"].dropna().tolist()
    
    total_queries = len(queries)
    logger.info(f"Found {total_queries} queries to process")
    
    # Determine embedding dimension
    embed_dim = model.get_sentence_embedding_dimension()
    logger.info(f"Embedding dimension: {embed_dim}")
    
    # Check for existing checkpoint
    checkpoint_file = f"{args.output_file}.checkpoint"
    processed_count = 0
    
    if os.path.exists(checkpoint_file):
        try:
            checkpoint_data = np.load(checkpoint_file, allow_pickle=True).item()
            processed_count = checkpoint_data.get('processed_count', 0)
            logger.info(f"Resuming from checkpoint. Already processed {processed_count} queries.")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting from the beginning.")
            processed_count = 0
    
    # Create memory-mapped file for results
    output_file = args.output_file.replace('.npy', '.memmap')
    try:
        if processed_count == 0 or not os.path.exists(output_file):
            memmap_file = np.memmap(
                output_file,
                dtype=np.float32,
                mode='w+',
                shape=(total_queries, embed_dim)
            )
        else:
            memmap_file = np.memmap(
                output_file,
                dtype=np.float32,
                mode='r+',
                shape=(total_queries, embed_dim)
            )
    except Exception as e:
        logger.error(f"Error creating memory-mapped file: {e}")
        return
    
    # Process queries in batches
    batch_size = args.batch_size
    
    try:
        for i in range(processed_count, total_queries, batch_size):
            batch_start_time = time.time()
            
            # Get current batch
            end_idx = min(i + batch_size, total_queries)
            batch_queries = queries[i:end_idx]
            current_batch_size = len(batch_queries)
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_queries+batch_size-1)//batch_size} ({i}-{end_idx-1})")
            
            try:
                batch_embeddings = model.encode(batch_queries, prompt_name="query", device=device)
                
                # Store in memory-mapped file
                memmap_file[i:end_idx] = batch_embeddings
                memmap_file.flush()
                
                # Clear batch from memory
                del batch_embeddings
                if device == "cuda":
                    torch.cuda.empty_cache()
                
                # Save checkpoint at intervals
                if (i + current_batch_size) % args.checkpoint_interval == 0 or end_idx == total_queries:
                    checkpoint = {'processed_count': end_idx}
                    np.save(checkpoint_file, checkpoint)
                    logger.info(f"Checkpoint saved at {end_idx} queries")
                
                batch_time = time.time() - batch_start_time
                queries_per_second = current_batch_size / batch_time
                logger.info(f"Batch processed in {batch_time:.2f}s ({queries_per_second:.2f} queries/second)")
                
                remaining_queries = total_queries - end_idx
                estimated_time = remaining_queries / queries_per_second if queries_per_second > 0 else 0
                logger.info(f"Progress: {end_idx}/{total_queries} ({end_idx/total_queries*100:.1f}%)")
                logger.info(f"Estimated time remaining: {str(timedelta(seconds=int(estimated_time)))}")
                
            except Exception as e:
                logger.error(f"Error processing batch {i}-{end_idx-1}: {e}")
                # Save checkpoint at point of failure
                checkpoint = {'processed_count': i}
                np.save(checkpoint_file, checkpoint)
                logger.info(f"Emergency checkpoint saved at {i} queries")
                # Try to continue with next batch
                continue
    
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
        checkpoint = {'processed_count': i}
        np.save(checkpoint_file, checkpoint)
        logger.info(f"Emergency checkpoint saved at {i} queries")
    
    finally:
        # Copy memory-mapped file to regular numpy array file
        if processed_count > 0:
            logger.info(f"Saving final embeddings to {args.output_file}")
            np.save(args.output_file, np.array(memmap_file))
            
            # Clean up checkpoint file if successful
            if os.path.exists(checkpoint_file) and processed_count == total_queries:
                os.remove(checkpoint_file)
                logger.info("Removed checkpoint file as all processing completed successfully")
    
    total_time = time.time() - start_time
    logger.info(f"Processing completed in {str(timedelta(seconds=int(total_time)))}")
    logger.info(f"Generated embeddings for {total_queries} queries and saved to {args.output_file}")

if __name__ == "__main__":
    main()
