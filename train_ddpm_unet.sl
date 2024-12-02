#!/bin/bash -e

#SBATCH --job-name=train-unet
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --time 00:15:00
#SBATCH --mem 20G
#SBATCH --qos=debug

module purge
source load-ml-environment.sh
export PYTHONNOUSERSITE=1

srun python train_ddpm_unet.py
