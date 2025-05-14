#!/bin/bash

#SBATCH -A bera89-k
#SBATCH --nodes=1 --gpus-per-node=1 
#SBATCH --cpus-per-task=32
#SBATCH --job-name dd100_simple
#SBATCH --constraint=K
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm_logs/dd100_simple.log
#SBATCH --error=slurm_logs/dd100_simple.err
#SBATCH --ntasks-per-node=1

# Load necessary modules (if required)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /depot/bera89/apps/dance

( while true; do nvidia-smi >> gpu_usage_train.log; sleep 60; done ) &
GPU_MONITOR_PID=$!

# Run the Python script
python tools/train_dd100.py

kill $GPU_MONITOR_PID
