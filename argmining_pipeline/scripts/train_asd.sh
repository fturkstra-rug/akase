#!/bin/bash
#SBATCH --job-name=train_asd_f1
#SBATCH --output=logs/train_asd_%j.log
#SBATCH --error=logs/train_asd_%j.err
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1

echo "Job started on $(hostname) at $(date)"

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

# Activate conda environment
source $HOME/venvs/mamkit/bin/activate

# Ensure logs directory exists
mkdir -p logs

# Run training
srun python $HOME/back-up/pipeline/train_asd_model.py \
     -i $HOME/back-up/pipeline/cross_topic_dataset \
     -o $HOME/back-up/pipeline/classification_output \
     --epochs 1

echo "Job finished at $(date) with exit code $?"
