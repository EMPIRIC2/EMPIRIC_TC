python slurm-launch.py \
  --exp-name test_data_generation \
  --command "python TrainingDataGeneration/GenerateTrainingData.py 10000 1 0 0 ./Data/ --on_slurm" \
  --load-env "source scripts/load-environment.sh" \
  --mem "1GB" \
  --time "00:15:00" \
  --chdir "/users/ewinkelm/EMPIRIC_AI_emulation/"
