#!/bin/bash
#SBATCH --job-name=find_neighbours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

export AWS_ACCESS_KEY_ID=#
export AWS_SECRET_ACCESS_KEY=#
export AWS_DEFAULT_REGION=#

source $HOME/venvs/mamkit/bin/activate
srun python $HOME/merge_subgraphs/find_neighbors.py
