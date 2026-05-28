#!/bin/bash
#SBATCH --job-name=drl_nofee
#SBATCH --account=25sfcs03
#SBATCH --partition=compute-normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
cd /lustrefs/project/25sfcs03/FIRST_TEST
source venv/bin/activate
export PYTHONPATH=/lustrefs/project/25sfcs03/FIRST_TEST:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=""
python scripts/train_no_fees.py
