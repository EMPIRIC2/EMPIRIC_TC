#!/bin/bash -e

#SBATCH --job-name=train-site-prob
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --time 02:00:00
#SBATCH --mem 8G

module purge
export CUDA_VISIBLE_DEVICES=''
source load-ml-environment.sh
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=''

srun python Machine\ Learning/conv_prob_predictor.py
