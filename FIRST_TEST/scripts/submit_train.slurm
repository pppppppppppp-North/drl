#!/bin/bash
#SBATCH --job-name=drl_trading
#SBATCH --account=25sfcs03
#SBATCH --partition=compute-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
cd /lustrefs/project/25sfcs03/FIRST_TEST
source venv/bin/activate
export PYTHONPATH=/lustrefs/project/25sfcs03/FIRST_TEST:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=""
echo "Starting training at $(date)"
python scripts/quick_demo.py --days 30 --timesteps 1000
echo "Training finished at $(date)"
