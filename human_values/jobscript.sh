#!/bin/bash
#SBATCH --job-name=hvi_graph_run_panda
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

export TRANSFORMERS_CACHE=/scratch/p317595/.cache

source /home4/p317595/miniconda3/etc/profile.d/conda.sh
conda activate human_values

python predict_optimised.py --sentences-dir datasets/panda --output-file graph_run_panda.tsv