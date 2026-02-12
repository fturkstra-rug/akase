#!/bin/bash
#SBATCH --job-name=finetune_bart_base
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

export TRANSFORMERS_CACHE=/scratch/p317595/.cache

source $HOME/venvs/main_env/bin/activate

python create_graph.py -d Education -s seed_data_panda.json -v graph_run_education.tsv -m bart-base