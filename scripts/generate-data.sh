
python slurm-launch.py \
  --exp-name data_generation \
  --command "python ../TrainingDataGeneration/GenerateTrainingData.py 1000 25 5 5 /nesi/project/uoa03669/ewin313/storm_data/v5 --include_grids" \
  --load-env "source load-environment.sh" \
  --mem "2GB" \
  --time "018:00:00" \





