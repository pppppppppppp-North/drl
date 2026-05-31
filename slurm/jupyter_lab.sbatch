#!/bin/bash
#SBATCH --job-name=jupyter_drl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=6:00:00
#SBATCH --partition=compute-devel
#SBATCH --account=25sfcs03
#SBATCH --output=logs/jupyter_%j.out
#SBATCH --error=logs/jupyter_%j.err

set -eo pipefail

PROJECT_DIR="/lustrefs/project/25sfcs03/drl_thai_stock"
ENV_DIR="/lustrefs/project/25sfcs03/envs/drl_env"
JUPYTER="$ENV_DIR/bin/jupyter"

cd "$PROJECT_DIR"
mkdir -p logs

test -x "$JUPYTER"

set -u

PORT=$((8800 + RANDOM % 1000))
HOST=$(hostname)

echo "JupyterLab node: $HOST"
echo "JupyterLab port: $PORT"
echo "SSH tunnel:"
echo "ssh -L ${PORT}:${HOST}:${PORT} s00658@hpc.jsite.me"

"$JUPYTER" lab --no-browser --ip=0.0.0.0 --port="$PORT"
