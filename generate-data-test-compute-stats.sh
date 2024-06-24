python slurm-launch.py \
  --exp-name test_data_generation \
  --command "python ./TrainingDataGeneration/GenerateTrainingData.py 10000 1 0 0 ./Data/run6 --include_grids --compute_stats --constant_historical_inputs" \
  --load-env "source load-environment.sh" \
  --mem "3GB" \
  --time "00:15:00" \
  --debug True
