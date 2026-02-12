"""
Argumentative sentence detection enrichment script (multiprocess single-node GPU/CPU)

Features:
- Parallel file-level processing using Python multiprocessing for a single SLURM node.
- One designated worker uses the GPU (recommended Option A). Other workers run on CPU to
  perform I/O, preprocessing and reduce wall time.
- Safe to interrupt and resume: each file is written to an `.inprogress` Parquet and atomically
  moved to final path when complete.
- Adds a new column `argumentative_probs` (list<float>) to each parquet file; original
  `preprocessed_sentences` remains untouched.

Usage example (SLURM, single node with GPU):
    srun -N1 -n1 --gres=gpu:1 --cpus-per-task=36 \
      python argumentative_sentence_enrichment.py \
        --input-dir /gpfs/scratch1/shared/fturkstra/.owi/public/main_clean \
        --output-dir /gpfs/scratch1/shared/fturkstra/.owi/public/main_enriched \
        --num-workers 6 --docs-batch 2000 --sent-batch-size 256

Notes:
- The GPU worker is process 0 (first spawned); it gets `CUDA_VISIBLE_DEVICES` left alone so
  PyTorch can use the GPU. CPU workers set `CUDA_VISIBLE_DEVICES=""` to avoid accidental GPU usage.
- The script assumes the model loading and prediction calls are thread/process-safe when instantiated
  per process (we create a separate SentenceDetector instance per worker process).

"""

import argparse
import os
import shutil
import logging
from pathlib import Path
from typing import List, Tuple
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm

# delayed import of model into worker processes
# from src.argument_mining.sentence_detection import SentenceDetector

LOG = logging.getLogger("argumentative_enricher")


def find_parquet_files(root: Path) -> List[Path]:
    files = [p for p in root.rglob("*.parquet") if p.is_file()]
    files.sort()
    return files


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def read_table_slice(parquet_file: pq.ParquetFile, start_row: int, num_rows: int) -> pa.Table:
    total_rows = parquet_file.metadata.num_rows
    if start_row >= total_rows:
        return pa.Table.from_batches([])

    remaining = min(num_rows, total_rows - start_row)
    batches = []

    row_group_count = parquet_file.num_row_groups
    cur = 0
    target_start = start_row
    target_end = start_row + remaining

    for rg in range(row_group_count):
        rg_meta = parquet_file.metadata.row_group(rg)
        rg_rows = rg_meta.num_rows
        rg_start = cur
        rg_end = cur + rg_rows
        cur = rg_end

        if rg_end <= target_start:
            continue
        if rg_start >= target_end:
            break

        slice_start = max(0, target_start - rg_start)
        slice_end = min(rg_rows, target_end - rg_start)
        length = slice_end - slice_start
        if length <= 0:
            continue

        table = parquet_file.read_row_group(rg, columns=None)
        table_slice = table.slice(slice_start, length)
        batches.append(table_slice.to_batches())

    flat = [b for group in batches for b in group]
    if not flat:
        return pa.Table.from_batches([])
    return pa.Table.from_batches(flat)


def extract_sentences_from_column(col_pylist: List) -> Tuple[List[str], List[int]]:
    flat = []
    lengths = []
    for doc in col_pylist:
        if not doc:
            lengths.append(0)
            continue
        lengths.append(len(doc))
        for item in doc:
            if item is None:
                flat.append("")
            else:
                flat.append(item[0])
    return flat, lengths


def reconstruct_probs_by_doc(probs_flat: List[float], lengths: List[int]) -> List[List[float]]:
    out = []
    idx = 0
    for l in lengths:
        if l == 0:
            out.append([])
            continue
        out.append(probs_flat[idx: idx + l])
        idx += l
    return out


def safe_move(src: Path, dst: Path):
    """Atomically move src to dst; overwrite dst if exists."""
    try:
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
    except Exception:
        # last-resort copy+remove
        shutil.copy2(str(src), str(dst))
        src.unlink()


def process_file(infile: Path, input_root: Path, output_root: Path, docs_batch: int, sent_batch_size: int,
                 use_gpu: bool, force: bool = False):
    """
    Process a single parquet file: read in chunks, predict sentence argumentative probs, write an
    .inprogress parquet and then atomically move to final output.
    This function is safe to call from multiple independent processes as long as they work on
    different files.
    """
    from argument_mining.sentence_detection import SentenceDetector
    LOG.info("Worker (GPU=%s) processing: %s", use_gpu, infile)

    rel = infile.relative_to(input_root)
    outpath = output_root.joinpath(rel)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    inprogress = outpath.with_suffix(outpath.suffix + ".inprogress")

    try:
        # quick skip if final exists and appears complete
        if outpath.exists() and not force:
            try:
                pqf_out = pq.ParquetFile(str(outpath))
                pqf_in = pq.ParquetFile(str(infile))
                if 'argumentative_probs' in pqf_out.schema.to_arrow_schema().names and pqf_out.metadata.num_rows == pqf_in.metadata.num_rows:
                    LOG.info("Skipping (already processed): %s", infile)
                    return
            except Exception:
                LOG.warning("Unable to validate existing output, will reprocess: %s", outpath)

        reader = pq.ParquetFile(str(infile))
        total_rows = reader.metadata.num_rows

        # determine resume start
        start_row = 0
        if inprogress.exists():
            try:
                pq_inprog = pq.ParquetFile(str(inprogress))
                start_row = pq_inprog.metadata.num_rows
                LOG.info("Resuming %s at row %d", infile, start_row)
            except Exception:
                LOG.warning("Corrupt inprogress file, restarting: %s", inprogress)
                inprogress.unlink()
                start_row = 0

        if start_row == 0 and inprogress.exists():
            try:
                inprogress.unlink()
            except Exception:
                pass

        # instantiate model inside worker process
        # make sure CUDA_VISIBLE_DEVICES was set before this function if GPU is desired
        asd_model = SentenceDetector()
        asd_model.load_or_train(force_train=False)

        cur_row = start_row
        pq_writer = None
        pbar = tqdm(total=total_rows, desc=f"{rel}", unit="rows")
        pbar.update(start_row)

        while cur_row < total_rows:
            to_read = min(docs_batch, total_rows - cur_row)
            table = read_table_slice(reader, cur_row, to_read)
            if table.num_rows == 0:
                break

            if 'preprocessed_sentences' not in table.schema.names:
                raise RuntimeError(f"preprocessed_sentences column not present in {infile}")

            col_pylist = table.column('preprocessed_sentences').to_pylist()
            flat_sents, lengths = extract_sentences_from_column(col_pylist)

            if len(flat_sents) == 0:
                probs_by_doc = [[] for _ in range(len(col_pylist))]
            else:
                _, probs = asd_model.predict(flat_sents, batch_size=sent_batch_size)

                probs_flat = []
                for p in probs:
                    if isinstance(p, (list, tuple)):
                        if len(p) == 1:
                            probs_flat.append(float(p[0]))
                        else:
                            probs_flat.append(float(p[1]))
                    else:
                        probs_flat.append(float(p))

                probs_by_doc = reconstruct_probs_by_doc(probs_flat, lengths)

            pa_probs = pa.array(probs_by_doc, type=pa.list_(pa.float64()))
            out_table = table.append_column('argumentative_probs', pa_probs)

            if pq_writer is None:
                pq_writer = pq.ParquetWriter(str(inprogress), out_table.schema)
            pq_writer.write_table(out_table)

            cur_row += out_table.num_rows
            pbar.update(out_table.num_rows)

        if pq_writer:
            pq_writer.close()

        safe_move(inprogress, outpath)
        LOG.info("Finished file %s -> %s", infile, outpath)
        pbar.close()

    except Exception as exc:
        LOG.exception("Error processing %s: %s", infile, exc)
        try:
            if pq_writer:
                pq_writer.close()
        except Exception:
            pass
        raise


def worker_main(worker_id: int, assigned_files: List[str], input_root: str, output_root: str,
                docs_batch: int, sent_batch_size: int, force: bool, use_gpu: bool):
    """
    Entrypoint for a worker process. `assigned_files` is a list of file paths (strings).
    `use_gpu` indicates whether this worker should use the GPU.
    """
    # Set CUDA device visibility before importing/initializing model
    if use_gpu:
        LOG.info("Worker %d enabling GPU", worker_id)
        # do not touch CUDA_VISIBLE_DEVICES to allow default GPU assignment
    else:
        LOG.info("Worker %d disabling GPU (CPU-only)", worker_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = ""

    # Configure per-worker logging to stdout (each worker writes to its own log file optionally)
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [worker {worker_id}] %(levelname)s %(message)s")

    input_root_p = Path(input_root)
    output_root_p = Path(output_root)

    for f in assigned_files:
        infile = Path(f)
        try:
            process_file(infile, input_root_p, output_root_p, docs_batch, sent_batch_size, use_gpu, force)
        except Exception:
            LOG.exception("Worker %d failed on file %s", worker_id, infile)
            # continue processing other files
            continue

def chunk_files_round_robin(files: List[Path], num_workers: int) -> List[List[str]]:
    batches = [[] for _ in range(num_workers)]
    for i, f in enumerate(files):
        batches[i % num_workers].append(str(f))
    return batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4, help="total number of processes to spawn")
    parser.add_argument("--docs-batch", type=int, default=2000,
                        help="number of documents (rows) to process in a single read/write batch")
    parser.add_argument("--sent-batch-size", type=int, default=None,
                        help="batch size passed to the model's predict() for sentences")
    parser.add_argument("--force", action="store_true", help="reprocess files even when outputs exist")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    ensure_parent(output_root)

    files = find_parquet_files(input_root)
    LOG.info("Found %d parquet files under %s", len(files), input_root)

    # Filter out files that appear already processed to reduce work dispatched
    remaining = []
    for f in files:
        rel = f.relative_to(input_root)
        outpath = output_root.joinpath(rel)
        if outpath.exists() and not args.force:
            try:
                pq_out = pq.ParquetFile(str(outpath))
                pq_in = pq.ParquetFile(str(f))
                if 'argumentative_probs' in pq_out.schema.to_arrow_schema().names and pq_out.metadata.num_rows == pq_in.metadata.num_rows:
                    continue
            except Exception:
                pass
        remaining.append(f)

    LOG.info("Remaining files to process: %d", len(remaining))
    if len(remaining) == 0:
        LOG.info("Nothing to do. Exiting.")
        return

    num_workers = min(args.num_workers, len(remaining))

    # Decide GPU worker assignment: worker 0 is GPU worker if CUDA available
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
    except Exception:
        cuda_avail = False

    if not cuda_avail:
        LOG.warning("CUDA not available; all workers will be CPU-only")
        gpu_worker_id = -1
    else:
        gpu_worker_id = 0

    batches = chunk_files_round_robin(remaining, num_workers)

    processes = []
    for wid in range(num_workers):
        assigned = batches[wid]
        use_gpu = (wid == gpu_worker_id)
        p = mp.Process(target=worker_main, args=(wid, assigned, str(input_root), str(output_root),
                                                 args.docs_batch, args.sent_batch_size, args.force, use_gpu))
        p.start()
        processes.append(p)
        LOG.info("Started worker %d (GPU=%s) with %d files", wid, use_gpu, len(assigned))

    # Wait for all workers
    for p in processes:
        p.join()

    LOG.info("All workers completed.")


if __name__ == '__main__':
    main()

