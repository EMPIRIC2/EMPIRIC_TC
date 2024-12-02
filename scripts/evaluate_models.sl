#!/bin/bash -e

#SBATCH --job-name=evaluate-models
#SBATCH --cpus-per-task=2
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time 00:15:00
#SBATCH --mem 20G
#SBATCH --chdir=/oscar/home/ewinkelm/EMPIRIC_AI_emulation

module purge
source activate data_env
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PYTHONPATH:/oscar/home/ewinkelm/EMPIRIC_AI_emulation

srun python MachineLearning/Evaluation/evaluate.py /oscar/home/ewinkelm/data/ewinkelm /oscar/home/ewinkelm/data/ewinkelm /oscar/home/ewinkelm/EMPIRIC_AI_emulation/Figures/AGU
