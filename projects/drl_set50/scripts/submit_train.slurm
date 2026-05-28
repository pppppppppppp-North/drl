#!/bin/bash
#SBATCH --job-name=drl_trading
#SBATCH --partition=gpu4500-normal
#SBATCH --account=25sfcs03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Load modules (Based on BistKA Policy)
module load Miniforge3/25.3.0-3

# Activate Environment (Ensure you created this on the cluster)
# conda create -n drl_env python=3.12 -y
# conda activate drl_env
# pip install -r requirements.txt
source ~/.bashrc
conda activate drl_env

# Run Training
echo "Starting training at $(date)"
python -m src.models.drl_agent --config configs/config.yaml

echo "Training finished at $(date)"
