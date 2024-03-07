
python slurm-launch.py \
  --exp-name data_generation \
  --command "python ./Training\ Data\ Generation/GenerateTrainingData.py 1000 25 5 5 /nesi/project/uoa03669/ewin313/storm_data/v3" \
  --load-env "source load-environment.sh" \
  --mem "1GB" \
  --time "06:00:00" \





