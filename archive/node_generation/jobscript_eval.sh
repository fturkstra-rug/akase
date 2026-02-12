#!/bin/bash
#SBATCH --job-name=eval_similarity
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

export HF_HOME=/scratch/p317595/.cache

source $HOME/venvs/main_env/bin/activate

python evaluate_similarity.py --labels labels_test_set.txt --predictions results_t5-base_test_set.txt --predictions_inverted results_t5-base_inverted_test_set.txt --model t5-base
python evaluate_similarity.py --labels labels_test_set.txt --predictions results_bart-base_test_set.txt --predictions_inverted results_bart-base_inverted_test_set.txt --model bart-base