#!/bin/bash
#SBATCH --job-name=drl_backtest
#SBATCH --account=25sfcs03
#SBATCH --partition=compute-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
cd /lustrefs/project/25sfcs03/FIRST_TEST
source venv/bin/activate
export PYTHONPATH=/lustrefs/project/25sfcs03/FIRST_TEST:$PYTHONPATH
echo "Starting backtest at $(date)"
python scripts/quick_backtest.py
echo "Backtest finished at $(date)"
