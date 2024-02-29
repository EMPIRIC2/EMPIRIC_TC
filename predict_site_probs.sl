#!/bin/bash -e

#SBATCH --job-name=predict-site-prob
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time 00:05:00
#SBATCH --mem 6G
#SBATCH --qos=debug

module purge
source load-ml-environment.sh
export PYTHONNOUSERSITE=1

srun python predict_site_probs.py





