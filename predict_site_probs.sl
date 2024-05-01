#!/bin/bash -e

#SBATCH --job-name=predict-site-prob
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --time 00:15:00
#SBATCH --mem 6G
#SBATCH --qos=debug

module purge
source load-ml-environment.sh
export PYTHONNOUSERSITE=1

srun python predict_site_probs.py





