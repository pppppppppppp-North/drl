#!/bin/bash
#SBATCH --job-name=cache_data
#SBATCH --account=25sfcs03
#SBATCH --partition=compute-devel
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:30:00
#SBATCH --output=%x_%j.out
cd /lustrefs/project/25sfcs03/FIRST_TEST
source venv/bin/activate
export PYTHONPATH=/lustrefs/project/25sfcs03/FIRST_TEST:$PYTHONPATH
python scripts/cache_data.py --tickers PTT.BK --start 2020-01-01 --end 2024-01-01
