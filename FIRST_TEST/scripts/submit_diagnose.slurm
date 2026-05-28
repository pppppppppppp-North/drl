#!/bin/bash
#SBATCH --job-name=diagnose
#SBATCH --account=25sfcs03
#SBATCH --partition=compute-devel
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:10:00
#SBATCH --output=%x_%j.out
cd /lustrefs/project/25sfcs03/FIRST_TEST
source venv/bin/activate
export PYTHONPATH=/lustrefs/project/25sfcs03/FIRST_TEST:$PYTHONPATH
python scripts/diagnose.py
