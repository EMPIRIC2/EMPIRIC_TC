
python slurm-launch.py \
  --exp-name data_generation_include_historical \
  --command "python TrainingDataGeneration/GenerateTrainingData.py 1000 0 25 0 /nesi/project/uoa03669/ewin313/storm_data/include_historical --include_grids --include_historical_genesis" \
  --load-env "source load-environment.sh" \
  --mem "2GB" \
  --time "18:00:00" \




