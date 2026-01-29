#!/bin/bash
#SBATCH --job-name=owilix-pull
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=owilix_download.log

export OWS_OWI_PATH=/gpfs/scratch1/shared/fturkstra/.owi

source /$HOME/download_owi/owi/bin/activate
yes | owilix remote pull all/collectionName=curlie_full files="**/language=eng/*"
