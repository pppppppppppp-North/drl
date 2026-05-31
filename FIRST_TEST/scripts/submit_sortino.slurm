#!/bin/bash
#SBATCH --job-name=drl_sortino
#SBATCH --account=25sfcs03
#SBATCH --partition=compute-normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
cd /lustrefs/project/25sfcs03/FIRST_TEST
source venv/bin/activate
export PYTHONPATH=/lustrefs/project/25sfcs03/FIRST_TEST:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=""
python scripts/train_sortino.py
