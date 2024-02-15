
python slurm-launch.py \
  --exp-name data_generation \
  --command "python ./Training\ Data\ Generation/GenerateTrainingData.py 1000 140 30 30 /nesi/nobackup/uoa03669/storm_data/v2" \
  --load-env "source load-environment.sh" \
  --mem "600MB" \
  --time "30:00:00" \





