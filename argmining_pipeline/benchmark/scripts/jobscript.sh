#!/bin/bash
#SBATCH --job-name=argmining_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --partition=gpu_a100
#SBATCH --time=02:00:00

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load FFmpeg/7.0.2-GCCcore-13.3.0

source $HOME/venvs/mamkit/bin/activate

# Process the downloaded benchmark, creates (sentences|components|relations).csv
srun python $HOME/argmining_benchmark/src/process_persuasive_data.py 
srun python $HOME/argmining_benchmark/src/process_toulmin_data.py

# Run the classifiers on the benchmark data and save the predictions.
srun python $HOME/argmining_benchmark/src/main.py -t asd -i ../data/sentences.csv -o ../data/sentences_predicted.csv
srun python $HOME/argmining_benchmark/src/main.py -t acc -i ../data/components.csv -o ../data/components_predicted.csv
srun python $HOME/argmining_benchmark/src/main.py -t arc -i ../data/relations.csv -o ../data/relations_predicted.csv

# Evaluate the results.
srun python $HOME/argmining_benchmark/src/eval.py -t asd -i ../data/sentences_predicted.csv
srun python $HOME/argmining_benchmark/src/eval.py -t acc -i ../data/components_predicted.csv
srun python $HOME/argmining_benchmark/src/eval.py -t arc -i ../data/relations_predicted.csv
