python slurm-launch.py \
  --exp-name grid_data_generation \
  --command "python ../TrainingDataGeneration/GenerateTrainingData.py 1000 1 0 0 /nesi/project/uoa03669/ewin313/storm_data/v4 10 100" \
  --load-env "source load-environment.sh" \
  --mem "1GB" \
  --time "06:00:00" \


