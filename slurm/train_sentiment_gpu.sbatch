#!/bin/bash
#SBATCH --job-name=thai_sent_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu4500-normal
#SBATCH --gres=gpu:pro4500:1
#SBATCH --account=25sfcs03
#SBATCH --output=logs/thai_sent_gpu_%j.out
#SBATCH --error=logs/thai_sent_gpu_%j.err

set -eo pipefail

PROJECT_DIR="/lustrefs/project/25sfcs03/drl_thai_stock"
ENV_DIR="/lustrefs/project/25sfcs03/envs/llm_env"
PYTHON="$ENV_DIR/bin/python"

cd "$PROJECT_DIR"
mkdir -p logs data/processed/sentiment models/sentiment

test -x "$PYTHON"

set -u

"$PYTHON" -m src.sentiment.extract_embeddings \
  --input data/processed/news_clean.parquet \
  --output data/processed/sentiment/wangchanberta_daily.parquet \
  --model airesearch/wangchanberta-base-att-spm-uncased \
  --batch-size 32
