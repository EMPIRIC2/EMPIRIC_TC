python slurm-launch.py \
  --exp-name test_data_generation \
  --command "python ./Training\ Data\ Generation/GenerateTrainingData.py 100 1 0 0 ./Data/run2" \
  --load-env "source load-environment.sh" \
  --mem "300MB" \
  --time "00:10:00" \
  --debug True
