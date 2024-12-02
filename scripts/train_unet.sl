#!/bin/bash -e

#SBATCH --job-name=train-unet
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --time 00:15:00
#SBATCH --mem 4G

module purge
source load-ml-environment.sh
export PYTHONNOUSERSITE=1

srun python train_unet.py
