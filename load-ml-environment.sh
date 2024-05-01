module purge
module load TensorFlow/2.13.0-gimkl-2022a-Python-3.11.3
module load cuDNN/8.9.7.29-CUDA-12.2.2

export PYTHONNOUSERSITE=1
source ml_env/bin/activate
