#!/bin/bash -e

#SBATCH --job-name=compute-train-mean
#SBATCH --cpus-per-task=2
#SBATCH --time 01:00:00
#SBATCH --mem 4G

module purge
source load-ml-environment.sh
export PYTHONNOUSERSITE=1

srun python ComputeTrainSampleMean.py
