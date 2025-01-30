#!/usr/bin/env bash
cd "$(dirname "$0")"
cd ..

conda deactivate
conda activate ml_env


PYTHONPATH="./":$PYTHONPATH python TrainingDataGeneration/GenerateTrainingData.py 10 1 0 0 ./example_code/example_data