#!/bin/bash
#SBATCH --job-name=asd_prediction
#SBATCH --output=logs/asd_prediction_%j.log
#SBATCH --error=logs/asd_prediction_%j.err
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

echo "Job started on $(hostname) at $(date)"

# Load modules if needed (example: conda & cuda)
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

# Activate your conda environment
source $HOME/venvs/mamkit/bin/activate

# Paths
INPUT_DIR=/gpfs/scratch1/shared/fturkstra/.owi/public/main_clean
OUTPUT_DIR=/gpfs/scratch1/shared/fturkstra/.owi/public/main_asd
MODEL_DIR=$HOME/back-up/pipeline/asd_output/final_model
SCRIPT_PATH=$HOME/back-up/pipeline/run_asd.py
LOG_DIR=logs

# Create log folder
mkdir -p $LOG_DIR

# Run the Python script
srun python $SCRIPT_PATH --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR --model_dir $MODEL_DIR --batch_size 12000

echo "Job finished at $(date) with exit code $?"
