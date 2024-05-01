#!/bin/bash -e

#SBATCH --job-name=train-site-prob
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=2
#SBATCH --time 00:15:00
#SBATCH --mem 4G
#SBATCH --qos=debug

module purge
source load-ml-environment.sh
export PYTHONNOUSERSITE=1

srun python train_site_probs.py
