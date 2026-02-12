#!/bin/bash
#SBATCH --job-name=sentence_segmentation_issues
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

export TRANSFORMERS_CACHE=/scratch/p317595/.cache

module --force purge
module load Python/3.11.5-GCCcore-13.2.0

source /home4/p317595/venvs/main_env/bin/activate

python prepare_input_hvi.py