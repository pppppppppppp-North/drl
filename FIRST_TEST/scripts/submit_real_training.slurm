#!/bin/bash
#SBATCH --job-name=drl_real
#SBATCH --account=25sfcs03
#SBATCH --partition=compute-normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
cd /lustrefs/project/25sfcs03/FIRST_TEST
source venv/bin/activate
export PYTHONPATH=/lustrefs/project/25sfcs03/FIRST_TEST:$PYTHONPATH
echo "Training with real data..."
python -m src.models.drl_agent --tickers PTT.BK --timesteps 50000
