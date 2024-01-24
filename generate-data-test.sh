python slurm-launch.py \
  --exp-name test_data_generation \
  --command "python ./Training\ Data\ Generation/GenerateInputParameters.py 5 1 1 ./Data" \
  --load-env load-environment.sh \
  --mem "512MB" \
  --time "00:03:00" \
  --debug True