#!/bin/bash -e

#SBATCH --job-name=train-unet
#SBATCH -p gpu --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time 02:00:00
#SBATCH --mem 10G

module purge
source load-pytorch-environment.sh
export PYTHONNOUSERSITE=1

srun python train_fno.py
