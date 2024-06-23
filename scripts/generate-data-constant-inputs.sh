python slurm-launch.py \
  --exp-name data_generation_constant_inputs \
  --command "python ../TrainingDataGeneration/GenerateTrainingData.py 1000 0 25 0 /nesi/project/uoa03669/ewin313/storm_data/constant_inputs 10 100 --constant_historical_inputs" \
  --load-env "source load-environment.sh" \
  --mem "2GB" \
  --time "18:00:00" \




