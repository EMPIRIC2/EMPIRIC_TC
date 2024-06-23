python slurm-launch.py \
  --exp-name test_data_generation \
  --command "python ../TrainingDataGeneration/GenerateTrainingData.py 100 1 0 0 ./Data/ 10 100" \
  --load-env "source load-environment.sh" \
  --mem "300MB" \
  --time "00:10:00" \
  --debug True
