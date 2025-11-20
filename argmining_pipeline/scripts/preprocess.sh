#!/bin/bash
#SBATCH --job-name=asd_preprocess
#SBATCH --partition=genoa           # CPU-only partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64          # cores for 1/8 node
#SBATCH --mem=336G                   # memory for 1/8 node
#SBATCH --time=04:00:00             # up to 2 days (adjust as needed)
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

echo "=============================================================="
echo "Job started on $(hostname) at $(date)"
echo "Running in: $(pwd)"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE"
echo "=============================================================="

# --- ENVIRONMENT SETUP ---
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

# Activate your virtual environment
source "$HOME/venvs/mamkit/bin/activate"

python -V

# --- CONFIGURATION ---
INPUT_DIR="/gpfs/scratch1/shared/fturkstra/.owi/public/main"
OUTPUT_DIR="/gpfs/scratch1/shared/fturkstra/.owi/public/main_clean"
SCRIPT_PATH="$HOME/back-up/pipeline/preprocess.py"

# Worker and batch size configuration
# WORKERS=6       
# BATCH_SIZE=5000   

# If you scale to a full node (336 GB), you can safely set:
WORKERS=12
BATCH_SIZE=500

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo "Starting preprocessing with $WORKERS workers and batch size $BATCH_SIZE"
echo "Input dir: $INPUT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "=============================================================="

# --- RUN ---
srun python "$SCRIPT_PATH" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --workers "$WORKERS" \
    --batch_size "$BATCH_SIZE"

EXIT_CODE=$?

echo "=============================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully."
else
    echo "Job failed with exit code $EXIT_CODE."
fi
echo "Finished at $(date)"
echo "=============================================================="
