#!/bin/bash -e

#SBATCH --job-name=evaluate-models-large-domain
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=2
#SBATCH --time 00:15:00
#SBATCH --mem 10G
#SBATCH --chdir=/nesi/project/uoa03669/ewin313/TropicalCycloneAI/
#SBATCH --qos=debug

module purge
source load-ml-environment.sh
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PYTHONPATH:/nesi/project/uoa03669/ewin313/TropicalCycloneAI/

srun python MachineLearning/Evaluation/evaluate.py /nesi/project/uoa03669/ewin313/storm_data/include_historical/ /nesi/project/uoa03669/ewin313/storm_data/v5/ /nesi/project/uoa03669/ewin313/TropicalCycloneAI/Figures/include_historical
