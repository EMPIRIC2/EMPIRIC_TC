#!/bin/bash -e

#SBATCH --job-name=train-unet
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time 00:10:00
#SBATCH --mem 4G

module purge
source load-ml-environment.sh
export PYTHONNOUSERSITE=1

srun python Machine\ Learning/evaluations.py


