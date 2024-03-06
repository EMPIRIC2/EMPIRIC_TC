#!/bin/bash -e

#SBATCH --job-name=train-site-prob
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --time 01:00:00
#SBATCH --mem 4G

module purge
source load-ml-environment.sh
export PYTHONNOUSERSITE=1

srun python train_site_probs.py
