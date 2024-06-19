python slurm-launch.py \
  --exp-name test_data_generation \
  --command "python ./TrainingDataGeneration/GenerateTrainingData.py 25 1 0 0 ./Data/run6 --include_grids --compute_stats --constant_historical_inputs" \
  --load-env "source load-environment.sh" \
  --mem "600MB" \
  --time "00:10:00" \
  --debug True
