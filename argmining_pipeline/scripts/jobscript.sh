#!/bin/bash
#SBATCH --job-name=run_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --partition=gpu_a100
#SBATCH --time=08:00:00

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load FFmpeg/7.0.2-GCCcore-13.3.0

source $HOME/venvs/mamkit/bin/activate
srun python $HOME/back-up/pipeline/run_pipeline.py
