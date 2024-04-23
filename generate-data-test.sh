python slurm-launch.py \
  --exp-name test_data_generation \
  --command "python ./Training\ Data\ Generation/GenerateTrainingData.py 200 1 0 0 ./Training\ Data\ Generation/data/ --include_grids" \
  --load-env "source load-environment.sh" \
  --mem "300MB" \
  --time "00:15:00" \
  --debug True
