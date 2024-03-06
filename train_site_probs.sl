#!/bin/bash -e

#SBATCH --job-name=train-site-prob
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time 00:40:00
#SBATCH --mem 4G

module purge
export CUDA_VISIBLE_DEVICES=''
source load-ml-environment.sh
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=''

srun python train_site_probs.py
