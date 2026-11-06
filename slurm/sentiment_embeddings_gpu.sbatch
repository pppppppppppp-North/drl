#!/bin/bash
#SBATCH --job-name=thai_sent_emb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu4500-normal
#SBATCH --gres=gpu:pro4500:1
#SBATCH --account=25sfcs03
#SBATCH --output=logs/thai_sent_emb_%j.out
#SBATCH --error=logs/thai_sent_emb_%j.err

set -eo pipefail

PROJECT_DIR="/lustrefs/project/25sfcs03/drl_thai_stock"
ENV_DIR="/lustrefs/project/25sfcs03/envs/llm_env"
PYTHON="$ENV_DIR/bin/python"

cd "$PROJECT_DIR"
mkdir -p logs data/processed/sentiment models/sentiment

test -x "$PYTHON"

set -u

# WangchanBERTa base has no sentiment head, so it is used for embeddings (PCA fitted on training
# dates only). For a sentiment score, rerun with --scorer classifier and a sentiment-fine-tuned model.
"$PYTHON" -m src.sentiment.score_news \
  --input data/processed/news_clean.parquet \
  --output data/processed/sentiment/wangchanberta_daily.parquet \
  --scorer embedding \
  --model airesearch/wangchanberta-base-att-spm-uncased \
  --fit-end-date "${FIT_END_DATE:-2021-12-31}" \
  --device cuda \
  --batch-size 32
