#!/bin/bash
#SBATCH --job-name=drl_long
#SBATCH --account=25sfcs03
#SBATCH --partition=compute-normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
cd /lustrefs/project/25sfcs03/FIRST_TEST
source venv/bin/activate
export PYTHONPATH=/lustrefs/project/25sfcs03/FIRST_TEST:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=""
echo "Starting LONGER training at $(date)"
# Train for 20,000 steps (20x longer than before)
# Use 100 days of data (more experience)
python scripts/quick_demo.py --days 100 --timesteps 20000
echo "Training finished at $(date)"
