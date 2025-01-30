#!/usr/bin/env bash
cd "$(dirname "$0")"
cd ..

conda deactivate
conda activate ml_env

data_path=./example_code/example_data
figures_path=./example_code/

PYTHONPATH="./":$PYTHONPATH python MachineLearning/Evaluation/evaluate.py $data_path $data_path $figures_path
