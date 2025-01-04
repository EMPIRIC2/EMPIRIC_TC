
python slurm-launch.py \
  --exp-name data_generation \
  --command "python TrainingDataGeneration/GenerateTrainingData.py 10000 0 10 0  /users/ewinkelm/data/ewinkelm/include_historical --include_historical_genesis --on_slurm" \
  --load-env "source scripts/load-environment.sh" \
  --mem "2GB" \
  --time "02:00:00" \
  --chdir "/users/ewinkelm/EMPIRIC_AI_emulation/"
  



