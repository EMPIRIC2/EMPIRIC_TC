python slurm-launch.py \
  --exp-name data_generation_stats \
  --command "python ./TrainingDataGeneration/GenerateTrainingData.py 10000 1 0 0 ./Data/stats_10000 --include_grids --compute_stats" \
  --load-env "source load-environment.sh" \
  --mem "2GB" \
  --time "40:00:00" \
  --debug True
